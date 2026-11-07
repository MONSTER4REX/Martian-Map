# P1 data / GIS layer

Owner: P1 (RTX 4050 laptop). The offline parts (derivatives, QA, error model)
run on any in-memory DEM and are tested on synthetic terrain. The network parts
(COG window reader, manifest against real buckets) are **UNTESTED ON REAL DATA**
until an owner runs them — the sandbox has no AWS/PDS access.

## Derivatives (`data/derivatives.py`)

NaN-aware; missing stays missing. Units: elevation m, cellsize m/cell.

- `slope_degrees(grid)` — terrain slope magnitude (deg).
- `aspect_degrees(grid)` — downslope compass aspect [0,360); NaN where flat.
- `hillshade(grid, azimuth_deg, altitude_deg)` — Horn hillshade 0..255.
- `curvature(grid)` — Laplacian curvature (1/m).

Convention: gradient from `numpy.gradient` × cellsize; aspect 0 = +row, toward
+col. Validated: inclined plane returns its slope; flat → aspect NaN and uniform
hillshade = cos(zenith)·255; plane curvature ≈ 0; a bump has negative (convex)
curvature; nodata propagates to neighbours.

## DTM QA (`data/qa.py`, C1)

`run_dtm_qa(grid, slope_spike_deg, step_m, seam_m)` → `(DTMQAReport, masks)`.
Flags (for inspection, **no elevations are modified**):

- no-data holes (count + fraction + mask),
- near-vertical slope spikes (processing artifacts),
- box-step discontinuities (large 2nd difference),
- CCD seam columns/rows (consistent offset along a line).

Thresholds are placeholders; tune per DTM. Flags feed the DEM error model (C2)
and the unknown mask — they are the §6 Mars-DTM pitfalls made measurable.

## DEM error ensembles (`data/error_model.py`, C2)

Spatially-correlated DEM error for route-stability (P6).

- `DEMErrorModel(sill_m2, range_m, nugget_m2)` — variance, correlation length,
  white component.
- `sample_error` / `perturbed_grid` / `ensemble(grid, model, n, seed)` —
  reproducible correlated error realizations; NaN cells stay NaN.
- `empirical_variogram(grid, …)` → `(lags_m, gamma)` isotropic semivariogram.
- `fit_exponential_variogram(lags, gamma)` → `DEMErrorModel` (grid-search range +
  non-negative LS for sill/nugget; no scipy).

Implementation note: the correlated field is a Gaussian random field (FFT
convolution of white noise with a Gaussian kernel), so its covariance is
approximately Gaussian, not a pure exponential variogram — fine for perturbation
ensembles. Owners fit `sill_m2`/`range_m` to **real** residuals (controlled-vs-
uncontrolled offsets, overlap differences) before trusting stability numbers.
Validated offline: sample variance ≈ sill; error decorrelates with distance;
ensembles reproducible and nodata-preserving; variogram rises then plateaus; the
fitter recovers exponential parameters on exponential data.
