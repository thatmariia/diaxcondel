# diaxcondel

A macroscopic model of EEG and MEG signals arising from white-matter
conduction delays.

The package is both a library and a dashboard. Its pipeline is short:

```text
distances (regions + lengths) + connectivity (weights)
    + delay kernel + local delay + noise + transfer
    → build_var()  → LinearVAR / NonlinearVAR   (Φ: P × N × N)
    → simulate()   → signals                    (T × N)
    → spectra, per-region features, event-related responses, sweeps
```

Geometry and coupling are chosen separately, so atlas distances can be paired
with a theoretical coupling rule, or hand-entered numbers with either.

```{toctree}
:caption: Using diaxcondel
:maxdepth: 1

using/installation
using/quickstart
using/model
using/atlas
using/experiments
using/playground
```

```{toctree}
:caption: Developing diaxcondel
:maxdepth: 1

developing/architecture
developing/components
developing/testing
```

```{toctree}
:caption: Reference
:maxdepth: 1

api
```
