# The model

Each region carries one number over time, the strength with which the rest of
the brain is driving it. For `N` regions and `P` lags,

```text
H_n(t) = Σ_{m≠n} Σ_p  γ[n,m] · w[n,m](p) · F(H_m(t − p·Δt))  +  ε_n(t)
```

where `γ` is the connectivity matrix, `w[n,m]` the distribution of
transmission delays along the connection from `m` to `n`, `F` an element-wise
transfer function (the identity in the published model), and `ε` the ongoing
noise. Matrix entry `[i, j]` is always input from `j` to `i`.

Because a signal that leaves a region comes back after a round trip through
the network, delays near `τ` reinforce frequencies near `1/(2τ)`. Which
rhythms a network produces is therefore a question about its geometry.

## Delays

A connection is a bundle of axons of different thicknesses, and thicker axons
conduct faster, so one length yields a distribution of arrival times rather
than a single delay. The package keeps the two ingredients as separate
choices, because they are separate measurements: the calibre distribution is
anatomical, and the constant relating calibre to velocity is physiological.

```python
spec = (
    ExperimentSpec()
    .with_component("diameter", "lognormal", median_um=0.3, sigma_log=0.5)
    .with_component("kernel", "hursh", speed_factor=6.0)
)
```

The `hursh` kernel applies `v = speed_factor · diameter` (Hursh, 1939), so a
connection of length `d` delivers its signal after `d/v`. Any calibre family
can be paired with any speed, and sweeping `kernel.speed_factor` moves every
delay without touching the anatomy. Four calibre distributions are available:
generalised extreme value (the one fitted in the paper), gamma, log-normal and
Rayleigh, all of which Sepehrband et al. (2016) fit to corpus-callosum
histology.

Two control kernels skip the calibre step. `gamma_delay` places a gamma spread
directly on delay, which is the neural-field convention; `fixed_speed` gives
every axon the same velocity. Neither declares a calibre distribution, so the
builder and the playground omit that slot when either is selected. A spectral
feature that survives all three constructions does not depend on how calibres
are distributed.

## Coupling

Peak sharpness is governed by the spectral radius of the assembled system, how
much of its own echo the network returns. Coupling strength, conduction speed
and geometry all move it at once, which makes it awkward to study on its own.
`target_spectral_radius` fixes it directly, scaling every coupling by a single
factor so that relative connectivity and every delay stay as they were:

```python
sweep(spec, {"target_spectral_radius": [0.85, 0.93, 0.97, 0.995]})
```

One is the stability boundary. Above it a linear model diverges, and only a
saturating transfer keeps activity bounded. A model whose coupling is spread
over many lags cannot be scaled far below one, since its radius falls only as
the `n_lags`-th root of the scale; the error names the reachable radius when a
target is out of range.

## Beyond the published model

Local delay
: A receiver-side delay (`local_delay` slot) convolved into every incoming
  connection, standing for postsynaptic rise and decay, layer crossing, and
  short fibres folded into the region. Gamma and biexponential forms are
  available. The original model omits this and absorbs local dynamics into the
  noise.

Recurrent self-excitation
: `self_weight` adds each region's own feedback through that local delay. It
  requires one, since instantaneous feedback cannot enter a VAR.

Saturating and rectifying transfer
: `centred_sigmoid` compresses smoothly, `hard_saturation` clips at a ceiling
  and is exactly linear below it, and `threshold_linear` rectifies. The
  rectifier is the one transfer that is not symmetric about zero, so activity
  acquires a mean and only its fluctuations are comparable with the linear
  model.

Shared input
: `noise:correlated` gives regions input they hold in common. Regions driven
  by the same signal are coherent without being connected, so this is the
  control for asking how much of a coherence pattern the connections explain.
  The closed-form spectrum survives it, since it requires the input to be
  white rather than independent.

Relayed paths
: `relay_code` routes every cortico-cortical connection through one region.
  Delays then add over two legs, and the builder selects the matching two-leg
  kernel. Travel time adds, so two legs average the same delay as a single hop
  of their combined length, but each leg draws its own axons and the arrival is
  more tightly timed.

## Spectra

For a linear model the power spectral density has a closed form,
`S(f) = T(f) Σ T(f)^H` with `T(f) = (I − Σ_p Φ_p e^{−2πifpΔt})^{-1}`. The noise
generators draw per-sample variance `std² · fs`, so the one-sided density is
`2 · std² · |T(f)|²` and the analytical and simulated spectra land on the same
axis at any sample rate.

The closed form needs a linear transfer and a white input. With a saturating
transfer or a sloped input spectrum it is omitted, and the reason is recorded
in the result. Welch and multitaper
estimators share the density convention, so all three can be compared
directly.
