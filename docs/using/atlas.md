# Atlas connectomes

With the `atlas` extra, geometry and coupling come from
[siibra](https://siibra-python.readthedocs.io):

```python
from diaxcondel.connectome.atlases.siibra_atlas import siibra_distances, siibra_weights

distances = siibra_distances("julich 3.1", merge_level=3)
weights = siibra_weights(distances, source="streamline_counts")
```

Streamline lengths become conduction distances in centimetres. Weights come
from streamline counts, functional connectivity, or a uniform placeholder.
Subject matrices are aggregated (or one subject selected), symmetrised, and
masked to the support of the length matrix, so a pair with no streamlines has
no connection.

`AtlasQuery` is recorded in the distance provider's metadata, which is how
`siibra_weights` replays exactly the same region pipeline and how the two
sources are guaranteed to describe the same regions.

## Choosing regions

Julich-Brain v3.1 has 414 regions. A VAR(300) over 414 nodes needs 400 MB for
the lag tensor alone and is neither tractable nor identifiable, so an atlas
model has to be coarsened first. There are two ways to say how.

`merge_level` cuts the atlas tree at one depth. For Julich-Brain, level 2
separates cortex from subcortical structures, level 3 gives lobes, level 4
gyri, level 5 cytoarchitectonic areas. `hierarchy_levels()` reports what each
level yields for a given parcellation.

`nodes` instead names the regions to keep, one by one:

```python
distances = siibra_distances("julich 3.1", nodes=("frontal lobe", "parietal lobe", "thalamus"))
distances.regions.codes      # ('parietal lobe', 'frontal lobe', 'thalamus')
```

One level rarely suits a whole brain. In Julich-Brain the lobes sit at level 3
while the thalamus is a level-2 region, so "a few lobes and the whole
thalamus" cannot be expressed as a cut of the tree at all. Anything not named,
or not inside something named, is left out. Appending a depth splits one entry
further while the others stay whole:

```python
distances = siibra_distances("julich 3.1", nodes=("frontal lobe:1", "thalamus"))
```

A name the atlas has matches only itself, so selecting the thalamus does not
also pull in the subthalamus. Free text that names no region falls back to the
highest region containing it, which is what makes `"occipital"` usable as a
search term.

`browse_regions()` lists the candidates with their level and how many parcels
each contains:

```python
from diaxcondel.connectome.grouping import browse_regions, describe_overlaps

native = siibra_distances("julich 3.1", merge_level=None, drop_isolated=False)
browse_regions(native.regions, level=3)              # the lobes and their peers
browse_regions(native.regions, contains="thalamus")  # across every level
```

Selections may overlap. `node_labels` gives a shared region to whichever entry
matches deepest in its ancestor chain, which is well defined but rarely what
someone means, so `describe_overlaps()` reports it:

```python
describe_overlaps(native.regions, ("cerebral cortex", "occipital lobe"))
# ("'cerebral cortex' contains 'occipital lobe' (20 regions in common); ...",)
```

The playground builds the same list as a table of selections, and shows
overlaps as warnings rather than errors.

## What merging does to delays

Coarsening an atlas turns one pathway into many connections of different
lengths. Distances are summarised by the plain unweighted mean, since that is
geometry. The *delay* of a merged pathway is treated differently: it is the
mixture of its members' delay distributions, weighted by how many streamlines
each carries.

```python
mixture = distances.mixture                        # member lengths and fibre counts
mixture.effective_lengths(len(distances.regions))  # fibre-weighted length per pair
```

Weighting a distance by fibre counts would be meaningless. Weighting how much
of the signal travels each distance is what a merged pathway is. The
difference matters: between frontal and occipital cortex the members carrying
most fibres are the long ones, and the fibre-weighted length comes out at
roughly twice the plain geometric mean. Set `use_length_mixture=False` to fall
back to a single representative length per connection.
