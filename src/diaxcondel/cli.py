"""
Command-line interface.

``diaxcondel`` exposes the same catalog and experiment machinery as the
package and the playground, so a configuration explored interactively can be
re-run on a cluster without touching Python:

.. code-block:: shell

    diaxcondel catalog                     # what components exist
    diaxcondel presets                     # ready-made experiments
    diaxcondel run spec.json -o run.npz    # run a saved specification
    diaxcondel sweep spec.json kernel.speed_factor --from 3 --to 9
    diaxcondel playground                  # open the dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from diaxcondel import __version__, catalog
from diaxcondel._cache import cache_dir, clear_cache
from diaxcondel.catalog.registry import SLOT_DESCRIPTIONS, SLOTS
from diaxcondel.experiment import (
    ExperimentSpec,
    build_model,
    compute_spectra,
    load_spec,
    parameter_paths,
    region_summaries,
    run_experiment,
    save_spec,
    sweep,
)
from diaxcondel.experiment.presets import get_preset, preset_names, preset_summary


def _print_catalog(slots: Sequence[str], as_json: bool) -> None:
    """Print the registered components, optionally as JSON."""
    if as_json:
        payload = {
            slot: [
                {
                    "name": entry.name,
                    "label": entry.label,
                    "summary": entry.summary,
                    "available": entry.available,
                    "requires": list(entry.requires),
                    "reference": entry.reference,
                    "params": {
                        field.name: {
                            "kind": field.kind,
                            "default": field.default,
                            "unit": field.unit,
                            "minimum": field.minimum,
                            "maximum": field.maximum,
                            "choices": list(field.choices) if field.choices else None,
                            "description": field.description,
                        }
                        for field in entry.fields
                    },
                }
                for entry in catalog.options(slot)  # type: ignore[arg-type]
            ]
            for slot in slots
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    for slot in slots:
        print(f"\n{slot}  —  {SLOT_DESCRIPTIONS.get(slot, '')}")  # type: ignore[arg-type]
        for entry in catalog.options(slot):  # type: ignore[arg-type]
            status = "" if entry.available else f"   [needs {', '.join(entry.requires)}]"
            print(f"  {entry.name:<16} {entry.label}{status}")
            if entry.summary:
                print(f"  {'':<16} {entry.summary}")
            for field in entry.fields:
                unit = f" {field.unit}" if field.unit else ""
                bounds = ""
                if field.minimum is not None and field.maximum is not None:
                    bounds = f" [{field.minimum}, {field.maximum}]"
                elif field.choices:
                    bounds = f" {{{', '.join(str(c) for c in field.choices)}}}"
                print(f"  {'':<18} - {field.name} = {field.default!r}{unit}{bounds}")


def _cmd_catalog(args: argparse.Namespace) -> int:
    """List catalog entries."""
    slots = [args.slot] if args.slot else list(SLOTS)
    _print_catalog(slots, args.json)
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    """List or export ready-made experiment specifications."""
    if args.name:
        spec = get_preset(args.name)
        if args.output:
            path = save_spec(spec, args.output)
            print(f"wrote {path}")
        else:
            print(spec.to_json())
        return 0
    for name in preset_names():
        print(f"{name:<22} {preset_summary(name)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a specification and optionally save the signals."""
    spec = load_spec(args.spec) if args.spec else ExperimentSpec()
    if args.trials is not None:
        spec = spec.model_copy(update={"n_trials": args.trials})
    if args.seconds is not None:
        spec = spec.with_simulation(sim_seconds=args.seconds)

    built = build_model(spec)
    result = run_experiment(spec, built=built, n_jobs=args.jobs)
    for note in result.all_notes:
        print(f"note: {note}", file=sys.stderr)

    spectra = compute_spectra(result)
    print(
        f"{spec.label or 'experiment'}: {built.n_regions} regions, {result.trials.n_trials} trial(s), "
        f"{result.signal.shape[0]} samples, spectral radius {built.spectral_radius():.3f}"
    )
    print(f"{'region':<24}{'alpha peak (Hz)':>16}{'1/f slope':>12}{'variance':>14}")
    for item in region_summaries(result, spectra):
        print(f"{item.code:<24}{item.peak_hz:>16.2f}{item.slope:>12.3f}{item.variance:>14.4g}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            trials=result.trials.trials,
            region_codes=np.array(result.regions.codes),
            sample_rate_hz=np.array(spec.simulation.sample_rate_hz),
            spec_json=np.array(spec.to_json()),
        )
        print(f"wrote {output}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Run one parameter over a range and report what changes."""
    spec = load_spec(args.spec) if args.spec else ExperimentSpec()
    available = parameter_paths(spec)
    if args.param not in available:
        print(f"unknown parameter {args.param!r}; available:", file=sys.stderr)
        for path in sorted(available):
            print(f"  {path}", file=sys.stderr)
        return 2

    values = list(np.linspace(args.start, args.stop, max(2, args.steps)))
    field = available[args.param]
    if field is not None and field.kind == "int":
        values = sorted({int(round(value)) for value in values})

    def report(index: int, total: int, point: dict) -> None:
        shown = ", ".join(f"{key}={value:g}" for key, value in point.items())
        print(f"[{index + 1}/{total}] {shown}", file=sys.stderr)

    result = sweep(spec, {args.param: values}, progress=report)
    rows = result.summary_table()
    header = list(rows[0])
    print("  ".join(f"{name:>16}" for name in header))
    for row in rows:
        cells = [f"{row[name]:>16.4g}" if isinstance(row[name], float) else f"{row[name]:>16}" for name in header]
        print("  ".join(cells))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        region_rows = result.region_table()
        columns = list(region_rows[0])
        lines = [",".join(columns)]
        lines += [",".join(str(row[name]) for name in columns) for row in region_rows]
        output.write_text("\n".join(lines))
        print(f"wrote {output}")
    return 0


def _cmd_playground(args: argparse.Namespace) -> int:
    """Start the interactive dashboard."""
    from diaxcondel.app import launch

    return launch(port=args.port, headless=args.headless)


def _cmd_cache(args: argparse.Namespace) -> int:
    """Show or clear the atlas download cache."""
    if args.clear:
        removed = clear_cache(args.namespace)
        print(f"removed {removed} cached file(s) from {cache_dir()}")
    else:
        print(cache_dir())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one sub-command per task.
    """
    parser = argparse.ArgumentParser(prog="diaxcondel", description=__doc__.splitlines()[1] if __doc__ else None)
    parser.add_argument("--version", action="version", version=f"diaxcondel {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="list available model components")
    catalog_parser.add_argument("slot", nargs="?", choices=list(SLOTS), help="restrict to one slot")
    catalog_parser.add_argument("--json", action="store_true", help="machine-readable output")
    catalog_parser.set_defaults(func=_cmd_catalog)

    presets_parser = subparsers.add_parser("presets", help="list or export ready-made experiments")
    presets_parser.add_argument("name", nargs="?", choices=list(preset_names()), help="preset to export")
    presets_parser.add_argument("-o", "--output", help="write the preset to this JSON file")
    presets_parser.set_defaults(func=_cmd_presets)

    run_parser = subparsers.add_parser("run", help="run an experiment specification")
    run_parser.add_argument(
        "spec",
        nargs="?",
        help="specification JSON file (defaults to the reference model)",
    )
    run_parser.add_argument("-o", "--output", help="save trials to this .npz file")
    run_parser.add_argument("--trials", type=int, help="override the trial count")
    run_parser.add_argument("--seconds", type=float, help="override the simulated duration")
    run_parser.add_argument("--jobs", type=int, default=1, help="worker processes for multi-trial runs")
    run_parser.set_defaults(func=_cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="vary one parameter and report what changes")
    sweep_parser.add_argument("spec", nargs="?", help="specification JSON file (defaults to the reference model)")
    sweep_parser.add_argument("param", help="parameter path, e.g. kernel.speed_factor")
    sweep_parser.add_argument("--from", dest="start", type=float, required=True, help="first value")
    sweep_parser.add_argument("--to", dest="stop", type=float, required=True, help="last value")
    sweep_parser.add_argument("--steps", type=int, default=7, help="number of values")
    sweep_parser.add_argument("-o", "--output", help="write per-region results to this CSV file")
    sweep_parser.set_defaults(func=_cmd_sweep)

    playground_parser = subparsers.add_parser("playground", help="open the interactive dashboard")
    playground_parser.add_argument("--port", type=int, help="port to serve on")
    playground_parser.add_argument("--headless", action="store_true", help="do not open a browser")
    playground_parser.set_defaults(func=_cmd_playground)

    cache_parser = subparsers.add_parser("cache", help="show or clear the atlas download cache")
    cache_parser.add_argument("--clear", action="store_true", help="delete cached downloads")
    cache_parser.add_argument("--namespace", help="only clear this sub-directory (e.g. siibra)")
    cache_parser.set_defaults(func=_cmd_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
