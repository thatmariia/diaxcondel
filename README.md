# diaxcondel

**diaxcondel** models **di**stributions of **ax**onal **con**nection **del**ays derived from inter-regional distances and axon-diameter distributions.

**Documentation: [thatmariia.github.io/diaxcondel](https://thatmariia.github.io/diaxcondel/)**

## Requirements

Python 3.12 or newer, with NumPy, SciPy, pydantic and platformdirs.
Optional dependencies:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `atlas` | siibra | atlas connectomes |
| `viz` | matplotlib | static figures |
| `app` | streamlit, matplotlib, plotly | the playground |
| `analysis` | specparam | separating periodic from aperiodic spectra |

## Quickstart playground

```bash
poetry install --with app,atlas,viz
poetry run diaxcondel playground
```

This opens a web-based playground for simulations.
From there, you can set the parameters in the **Model** tab and investigate its
behavior in other tabs. You can also perform a sweep by varying one parameter
over a range and plot a measure against it.

Every control explains what it does to the model, and every figure states how
it was produced. See [the playground
guide](https://thatmariia.github.io/diaxcondel/using/playground.html) for the
conventions.

## As a library

```python
from diaxcondel.experiment import ExperimentSpec, run_experiment, compute_spectra, region_summaries

result = run_experiment(ExperimentSpec())
for region in region_summaries(result, compute_spectra(result)):
    print(region.code, round(region.peak_hz, 2))
```

An `ExperimentSpec` validates against the component catalog as it is built,
serialises to JSON, and carries its seed, so it is a complete record of a run.
Building models, sweeping parameters, atlas connectomes and the component
catalog are covered in the
[documentation](https://thatmariia.github.io/diaxcondel/using/quickstart.html).

## Development

```bash
poetry run pytest                 # atlas downloads are skipped
poetry run pytest --run-network   # include them
poetry run ruff check             # lint
poetry run ruff format            # format
poetry run sphinx-build -b html docs docs/_build/html
```

See the [developer
documentation](https://thatmariia.github.io/diaxcondel/developing/architecture.html)
for the architecture and the conventions.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
