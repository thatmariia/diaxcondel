# Installation

## Requirements

Python 3.12 or newer. The core depends on NumPy, SciPy, pydantic and
platformdirs.
Optional dependencies:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `atlas` | siibra | atlas connectomes from siibra |
| `viz` | matplotlib | static figures |
| `app` | streamlit, matplotlib, plotly | the playground |
| `analysis` | specparam | separating periodic from aperiodic spectra |
| `docs` | sphinx, furo, myst-parser | building this documentation |

## Install

```bash
poetry install
```

Optional groups are opt-in:

```bash
poetry install --with atlas,viz,app
```

## Checking the install

```bash
poetry run diaxcondel catalog          # lists every model component
poetry run pytest                      # tests; atlas downloads are skipped
poetry run pytest --run-network        # include the live atlas downloads
```

Atlas downloads are cached on disk. `diaxcondel cache` prints the location,
which follows the platform convention unless `DIAXCONDEL_CACHE_DIR` overrides
it; `DIAXCONDEL_NO_CACHE=1` bypasses it altogether.
