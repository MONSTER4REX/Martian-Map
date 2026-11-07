# Science layers: size-frequency (C7) + value raster (C9)

Status: **offline baseline complete**, input-agnostic. Owners: P2/P6. Module:
`src/marswalk/science/`.

## Rock size-frequency (`size_frequency.py`, C7)

`fit_exponential_sfd(diameters_m, area_m2)` → `SFDFit` for the exponential
cumulative *number* density `N(>=D)/area = N0·exp(-λD)` (log-linear regression,
with R²). Helpers: `cumulative_number_density(fit, D)`,
`expected_count_at_least(fit, D, area)`.

**Caveat:** this is a generic, checkable fit, **not** the site-calibrated
Golombek & Rapp cumulative-fractional-*area* model `CFA(D)=k·exp(-q(k)·D)` that
is the Mars standard. An owner must calibrate that (against real counts and the
pretrained rock-abundance prior) before rock SFD drives routing. Validated:
recovers the decay rate of synthetic exponential samples (R² > 0.95); expected
count scales with area.

## Science-value raster (`value_raster.py`, C9)

`science_value({name: (raster, weight)})` → value raster in [0,1]: each input is
min-max normalized, then blended by the weights of the layers finite at each
cell (missing layers don't contribute; a cell is NaN only where all are NaN).
`top_targets(value, k, min_separation_cells)` → top-k cells (greedy spacing) to
feed the orienteering planner as `ScienceTarget`s.

Inputs (mineralogy, geologic units, feature proximity) come from P1/P2 later;
this module invents no science values. Validated: weighted blend arithmetic,
NaN handling, and spacing-constrained selection.
