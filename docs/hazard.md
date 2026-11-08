# Rock-hazard layer + the §3 ML gate

The link from the rock detector (P2) to the planner (P3), and the ablation that
decides whether the rock layer ships. Modules: `planner/hazard.py`,
`planner/hazard_routes.py`, `validation/ablation.py` (`rock_ablation`).

## Why

MARSWALK_CONTEXT.md §3 gates the rock detector: it informs routing **only if an
ablation shows rock-aware routes reduce rock exposure vs slope-only**. Before
this, the detector produced rocks but nothing consumed them and nothing tested
the gate. Now it does.

## Pieces

- **`hazard.rock_hazard_field(grid, rocks, min_diameter_m=1.5, buffer_m=2.0)`** —
  turns detected `Rock`s (grid position + diameter) into a per-cell hazard field
  in [0, 1]. Only rocks ≥ `min_diameter_m` count (the gate size is 1.5 m); each
  paints a footprint disk decaying over a `buffer_m` clearance margin; cells take
  the max over nearby rocks. `density_to_hazard` does the same from a
  boulder-density raster. The layer carries **no slope information** — kept
  separable so the ablation isolates the rock effect.
- **`hazard.rock_exposure(grid, hazard, path)`** — a route's integrated rock
  hazard (hazard·m): mean endpoint hazard × edge length, summed.
- **`hazard_routes.plan_hazard_route(grid, hazard, start, goal, cfg, beta, max_hazard)`**
  — full-constraint A* with edge cost `× (1 + beta·edge_hazard)`. `beta=0`
  reproduces the slope-only mission route exactly; larger `beta` trades distance
  for lower rock exposure; `max_hazard` hard-excludes an impassable field. The
  hazard multiplier is ≥ 1, so the straight-line heuristic stays admissible.
- **`validation.ablation.rock_ablation(grid, hazard, start, goal, betas=…)`** —
  plans at each `beta` (0 = baseline), measures rock exposure on the *same*
  hazard field, and reports `gate_passed` (rock-aware both lowered exposure and
  changed the route) with the exposure-reduction fraction and a verdict.

## Demo on real Mars boulders

`scripts/rock_ablation_demo.py` runs the gate on a **real** boulder field —
461 boulders from HiRISE `ESP_017355_2260` (BOULDERING, Zenodo 8171052,
CC BY 4.0), stored in `examples/bouldering_ESP_017355_2260.json` (real centres +
diameters). On a demo AOI the slope-only route plows through boulder clusters
(rock exposure ≈ 38); the rock-aware route detours through the gaps (≈ 1, −98%)
— **gate passed**. Figure: `docs/rock_ablation.png`.

```bash
python scripts/rock_ablation_demo.py          # needs numpy; matplotlib for the PNG
```

## Validity

The hazard **magnitudes/thresholds are PLACEHOLDERS** and the demo AOI is a
demo scale (1 m/cell) — absolute metres need the HiRISE product GSD, which an
owner sets. The boulder *positions/sizes* are real. On real data an owner feeds
georeferenced detections (from `train_rock_detector.ipynb`) as `Rock`s or a
density raster; the mechanism and the gate test are what this provides.
