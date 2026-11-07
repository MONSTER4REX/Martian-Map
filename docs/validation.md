# P6 validation: metrics, route stability, ablation

Status: **offline baseline complete.** Owner: P6. Module:
`src/marswalk/validation/`.

## Route metrics (`metrics.py`)

- `jaccard(a, b)` — cell-set similarity of two routes [0,1].
- `path_length_3d(grid, path)` — true 3-D length (m).
- `hazard_exposure(grid, path, max_slope_deg)` — count of steep edges.
- `discrete_frechet(grid, p, q)` — Fréchet distance (m) between two routes
  (iterative DP). Used for Perseverance-consistency comparisons later.

## Route stability (`stability.py`) — headline uncertainty figure

`route_stability(grid, start, goal, model, n, plan, pcfg, seed)` replans the
route across a DEM error ensemble (C2) and reports:

- `frac_identical` — fraction of realizations whose route equals the baseline,
- `mean_jaccard` — mean cell-set similarity to the baseline,
- `frequency` — per-cell fraction of realizations whose route passes through it
  (this is the map you render as the stability heat layer),
- feasible/infeasible counts.

A low `frac_identical` / `mean_jaccard` means the recommendation is driven by
DEM error and must be shown as uncertain, never as a single certain route.
Validated: negligible error → identical routes; realistic error → similarities
stay in [0,1] and the frequency map is a proper fraction.

## Ablation (`ablation.py`) — the feature gate

`ablation(grid, start, goal, configs, baseline, policy)` plans each named config
and flags those whose route differs from the baseline's (`changed_vs_baseline`,
via a Jaccard threshold), with per-config length and hazard exposure. This is
how a feature earns its place: it ships only if toggling it measurably changes
the route (`report.changed()`). Validated: a slope-limit change that forces a
ridge detour is flagged as changed; an identical config is not.

## Still to come (P6 scope)

Perseverance held-out consistency (Fréchet + hazard on PLACES segments),
negative cases at other rover sites, thermal-vs-MEDA, SPICE geometry tests,
second-AOI transfer — these need real data and owners' machines.
