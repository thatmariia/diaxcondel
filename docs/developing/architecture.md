# Architecture

The package has three layers. Code that faces a user should prefer the higher
two.

```text
catalog          named components, each describing its own options
  ↓
ExperimentSpec   a JSON-serialisable description of one run
  ↓
build_model() / run_experiment() → BuiltModel / RunResult
```

Underneath sits the numerical core: providers for distances and connectivity,
a delay kernel, and `build_var()`, which turns them into a lag tensor.

## Modules

`model`
: VAR dynamics. `build_var()` assembles the `(P, N, N)` lag tensor and returns
  `LinearVAR` or `NonlinearVAR`. `stability.py` answers stationarity two ways:
  dense eigenvalues for small systems, and a winding-number count of
  `det(I − Σ Φ_p z^p)` above `DENSE_EIGENVALUE_LIMIT`. Both give the same
  answer; the second is about 300 times faster at atlas scale.
  `scale_to_spectral_radius()` places a model at a chosen distance from
  criticality by bisection.

`connectome`
: Protocol-based providers for regions, distances and weights, bundled by
  `Connectome`. `from_sources()` checks that a distance source and a
  connectivity source describe the same regions. `grouping.py` holds the two
  ways of coarsening an atlas, `ancestor_labels()` and `node_labels()`, plus
  `browse_regions()` and `describe_overlaps()`. `atlases/siibra_atlas.py` is
  the siibra adapter.

`catalog`
: The library of named components. See [adding components](components.md).

`experiment`
: `ExperimentSpec` and the functions that build, run, report and sweep it.

`kernels`
: Connection length in centimetres to a delay distribution in seconds.
  `diameter.py` is the core: a `DiameterDistribution` protocol, a
  `DiameterDelayKernel` over any of its members, and `TwoLegDelayKernel` for
  relayed paths.

`simulate`
: The linear path uses a sliding-window state-space update, one `(N, NP)`
  matrix-vector product per sample, and never forms the `(NP, NP)` companion
  matrix. A nonlinear transfer falls back to a per-step loop.

`spectral`, `analysis`
: Closed-form and empirical spectra sharing one density convention, and the
  post-hoc summaries built on them.

`viz`, `app`
: Static figures, and the Streamlit playground over them. Presentation only.

`inference`
: Early scaffolding for parameter recovery: `ParameterSpec`, transforms, and a
  Whittle-style likelihood.

## Working at atlas scale

Fine parcellations are the reason several parts of the package look the way
they do. A 314-region VAR(300) needs 230 MB for `Φ`, a 94 200 × 94 200
companion matrix, and hours of simulation. So models are coarsened first,
`estimate_cost()` reports the cost before anything is built, and neither the
stationarity test nor the simulation forms the companion matrix.

## Design patterns

Protocols
: `Dynamics`, `DelayKernel`, `DistanceProvider`, `ConnectivityProvider` and
  `Drive` are `typing.Protocol`s, so any conforming object works and no
  inheritance is required.

Derived metadata
: A component's options come from its signature, annotations and docstring.
  Adding a parameter is enough for it to appear in the CLI listing, the
  dashboard and spec validation.

Provenance over silence
: Builders capture the warnings they raise (stationarity shrinkage, atlas
  aggregation choices, coupling rescaling) and return them.

Explicit randomness
: Every stochastic function takes an `rng`, which may be a seed or a
  `numpy.random.Generator`. There is no hidden global state.

## Scientific guardrails

- Do not normalise away physically meaningful scale unless the function name
  and docstring say so.
- Keep units in names and docstrings (`cm`, `s`, `Hz`, `ms`), and validate
  conversions at boundaries.
- Matrix entry `[i, j]` is input from source `j` to target `i`.
- If a stability fix rescales `Φ`, report it. A change in effective coupling
  must not be hidden.
- A connection that carries weight must carry a delay. Where a merged pathway
  has coupling but no measured member length, the builder raises rather than
  dropping the connection quietly.
- Closed-form spectra require an identity transfer and a white input.
