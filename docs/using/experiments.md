# Experiments

## Specifications

`ExperimentSpec` describes one model and one simulation. It names a catalog
entry per slot, carries the simulation settings, the seed, any events, and the
few model-level options that do not belong to a component (`self_weight`,
`relay_code`, `target_spectral_radius`, `include_regions`).

```python
from diaxcondel.experiment import ExperimentSpec, load_spec, save_spec

spec = ExperimentSpec(label="resting state")
save_spec(spec, "spec.json")
assert load_spec("spec.json") == spec
```

Validation happens on construction. `spec.fingerprint()` gives a stable hash
of the whole configuration, which is convenient for caching and for naming
output files.

## Running

```python
from diaxcondel.experiment import build_model, run_experiment

built = build_model(spec)       # assembles the lag tensor, no simulation yet
built.spectral_radius()
built.notes                     # warnings captured during construction
built.cost()                    # memory and runtime estimate

result = run_experiment(spec)   # builds and simulates
```

`build_model` captures the warnings raised while assembling the model and
returns them as `notes` instead of letting them scroll past. The most
important is the stationarity shrink, which changes effective coupling and so
has to be reported.

`estimate_cost` answers how expensive a configuration is before anything is
built. The playground refuses heavy configurations until confirmed.

## Reporting

```python
from diaxcondel.experiment import compute_spectra, delay_statistics, region_summaries

spectra = compute_spectra(result, estimator="multitaper")
summaries = region_summaries(result, spectra)
delays = delay_statistics(built)
delays.describe()   # mean, shortest and longest delay, and the frequencies they imply
```

`compute_spectra` returns the simulated spectrum and, where one exists, the
closed form on the same scale, with a note when it does not.
`region_summaries` gives the alpha-band peak, the log-log slope, the aperiodic
exponent and the time-domain variance for each region. The exponent comes from
[specparam](https://specparam-tools.github.io) when it is installed, and from
a straight-line fit otherwise; the route taken is reported either way.

## Sweeps

```python
from diaxcondel.experiment import parameter_paths, sweep

parameter_paths(spec)                                        # what this setup offers
sweep(spec, {"kernel.speed_factor": [4.0, 6.0, 8.0]})        # one axis
sweep(spec, {"kernel.speed_factor": [4.0, 8.0],              # a grid
             "connectivity.coupling": [0.6, 0.9]})
```

The seed stays fixed across a sweep, so differences come from the parameter
rather than from the noise.

## Events

An event adds a stimulus, a temporary change of coupling, or both, at a chosen
time in every trial.

```python
spec = ExperimentSpec(n_trials=20).with_event("gaussian_pulse", onset_seconds=0.5, targets=("OL",))
result = run_experiment(spec)
result.average          # trial average
result.trials.sem       # standard error across trials
```

Available waveforms are a Gaussian pulse, a square pulse, a sine burst, a
frequency sweep (`chirp`) and an arbitrary sampled waveform. A `step_gain`
modulation scales incoming connections for a window; the schedule it produces
is piecewise constant, so a modulation costs one stability check and one lag
tensor per stretch rather than one per sample.

## Presets and the command line

```bash
poetry run diaxcondel presets                                   # ready-made experiments
poetry run diaxcondel presets atlas_lobes -o spec.json
poetry run diaxcondel run spec.json -o run.npz
poetry run diaxcondel sweep spec.json kernel.speed_factor --from 3 --to 9 --steps 7
poetry run diaxcondel catalog --json                            # machine-readable component list
```

Presets are ordinary specifications and a reasonable place to start reading:
`reference_resting` is the published model, `criticality_sweep` sits just below
the stability boundary, `shared_input_control` has no coupling at all, and
`frequency_sweep` drives one region across 2–40 Hz.
