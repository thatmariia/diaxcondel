# Adding a component

A component is a factory function with a decorator. Its options come from the
signature, the type annotations, the `Param` metadata and the NumPy docstring,
so nothing has to be registered twice.

```python
from typing import Annotated
from diaxcondel.catalog import Param, register


@register("kernel", "my_kernel", label="My kernel", tags=("alternative",))
def my_kernel(
    speed_m_s: Annotated[float, Param(unit="m/s", minimum=0.5, maximum=30.0)] = 6.0,
) -> DelayKernel:
    """
    One-line summary, shown in menus.

    A paragraph on what the component does to the model. Write it for someone
    who has not read the paper.

    Parameters
    ----------
    speed_m_s : float
        Conduction speed. This text becomes the control's help.

    Returns
    -------
    DelayKernel
        What the factory produces.
    """
    ...
```

That is enough for the entry to appear in `diaxcondel catalog`, in the
playground with a labelled control of the right type and range, and in
`ExperimentSpec` validation.

## Slots

| Slot | Provides |
| --- | --- |
| `distances` | regions and their separations (cm) |
| `connectivity` | coupling weights over those regions |
| `diameter` | axon calibre distribution (µm) |
| `kernel` | length to a distribution of transmission delays |
| `local_delay` | receiver-side delay |
| `noise` | the ongoing stochastic input |
| `transfer` | element-wise output transfer |
| `drive` | stimulus waveform for events |
| `modulation` | time-limited coupling change |

`SLOT_INFO` carries each slot's one-line summary for the command line and its
longer description for the dashboard. Both come from one definition.

## Build context

A parameter without a default is build-time *context* rather than a user
option. The builder supplies it. This is how a connectivity rule sees the
geometry it depends on:

```python
@register("connectivity", "distance_decay")
def distance_decay(regions, distances, length_scale_cm: float = 8.0): ...
```

and how a calibre-based kernel receives its diameter distribution. A kernel
that sets speed directly declares no `diameter` context, so the builder and
the playground both omit that slot for it.

## Optional dependencies

`@register(..., requires=("siibra",))` keeps an entry listed when its
dependency is missing and reports an install hint instead of failing at
import.

## Widgets

`Param(widget=...)` tells the dashboard how to render a value it could not
infer: `"matrix"` and `"vector"` for editable tables, `"regions"` for a region
picker, `"hidden"` for an option the page renders with a control of its own.
`Param(advanced=True)` moves a secondary option behind an expander.
