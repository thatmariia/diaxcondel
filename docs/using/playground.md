# The playground

```bash
poetry run diaxcondel playground
```

The dashboard is a front end over `ExperimentSpec`. It holds no modelling
logic, so a component added to the catalog appears in it without any change to
the page.

## Layout

**Model** has one sub-tab per concern: regions and distances, connections,
delays, local processing, and input and output. **Network** shows a
distance-faithful layout, the matrices, the connection lengths and the delays
they produce. **Signals**, **Spectra** and **Events** show what the model did.
**Sweep** varies one parameter and plots a measure against it. **Guide**
explains the model and lists every option the catalog offers. **Reproduce**
hands back the specification and a runnable script.

## Conventions

Every figure carries a note saying how it was produced: which estimator, which
window, how many trials, how much display smoothing. Smoothing controls are
display-only and say so, including the frequency they cut.

Options that cannot describe the current model stay in their menus, marked
and explained. Knowing that a coupling rule exists but does not fit the
current regions is more useful than never seeing it.

Parameter help comes from the entry docstring, and the slot descriptions come
from the catalog, so the page cannot drift away from what the package does.

## Figures

Figures are interactive when plotly is installed: hover for the exact value,
drag to zoom into a band, click a legend entry to drop a region. Without
plotly the page falls back to the static matplotlib figures.

## Choosing atlas regions

The regions sub-tab describes an atlas model as a table of selections. Each
row picks regions and says how far to split them, and the model is every row
together, so one row can hold a few lobes kept whole while the next splits one
branch further. The level filter on a row only shortens the list beside it.
Overlapping selections produce a warning that says which entry keeps the
regions they share.
