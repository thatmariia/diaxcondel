# Quickstart

## Run the reference model

```python
from diaxcondel.experiment import ExperimentSpec, run_experiment, compute_spectra, region_summaries

spec = ExperimentSpec()                      # the five-region model of the paper
result = run_experiment(spec)
spectra = compute_spectra(result)

for region in region_summaries(result, spectra):
    print(region.code, round(region.peak_hz, 2), round(region.exponent, 2))
```

`ExperimentSpec` names one component per slot and validates against the
catalog as it is constructed, so a misspelled entry or an out-of-range
parameter fails immediately rather than at simulation time. It serialises to
JSON and carries the random seed, which makes it a complete record of a run.

## Change a component

```python
spec = (
    ExperimentSpec()
    .with_component("distances", "siibra", parcellation="julich 3.1", merge_level=3)
    .with_component("connectivity", "siibra", source="streamline_counts")
    .with_component("local_delay", "gamma", mean_ms=3.0)
    .with_simulation(sample_rate_hz=500, n_lags=150)
)
```

## Sweep a parameter

```python
from diaxcondel.experiment import sweep

result = sweep(spec, {"kernel.speed_factor": [3.0, 4.5, 6.0, 7.5, 9.0]})
result.summary_table()     # peak frequency, slope, spectral radius per value
result.region_table()      # the same, per region
```

Any numeric parameter has a path: `"connectivity.coupling"`,
`"simulation.sample_rate_hz"`, `"self_weight"`. `parameter_paths(spec)` lists
what a given setup offers.

## Build the pieces

The declarative route above is a wrapper over the underlying objects, which
can be assembled directly:

```python
from diaxcondel.catalog import build
from diaxcondel.model import SimulationParams, build_var
from diaxcondel.simulate import get_noise_fn, simulate

distances = build("distances", "steeghs_2025")
weights = build("connectivity", "uniform", {"coupling": 0.9},
                regions=distances.regions, distances=distances)
calibres = build("diameter", "gev")
kernel = build("kernel", "hursh", {"speed_factor": 6.0}, diameter=calibres)
params = SimulationParams(sample_rate_hz=1000, sim_seconds=10.0, burnin_seconds=1.0, n_lags=300)

model = build_var(distances, weights, kernel, params)
result = simulate(model, params, get_noise_fn("white"), rng=20260505, regions=distances.regions)
```

## Open the playground

```bash
poetry run diaxcondel playground
```

See [the playground](playground.md) for what it offers.
