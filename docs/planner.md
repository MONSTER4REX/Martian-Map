# P3 Planner baseline

Status: **offline baseline complete, UNTESTED ON REAL DATA.** Runs on synthetic
terrain only (the sandbox has no AWS/PDS/GPU access). Owner: P3.

## What it does

Plans three routes to one destination and explains why they differ:

| Policy | Edge cost | Hard exclusions |
|---|---|---|
| `shortest` | true 3-D segment length | unknown cells only |
| `slope_aware` | 3-D length × slope penalty (≥1) | unknown cells only |
| `full_constraint` | 3-D length × slope penalty | unknown + along-slope + cross-slope |

Each route reports length, cell count, max/mean along-slope, max cross-slope,
hard-exclusion violation count, mission-feasibility, and a plain-language reason
when it is rejected for mission use.

## Inputs / outputs / units

- **Input:** `Grid(elevation[nrows,ncols] meters, cellsize meters)`. `NaN` =
  unknown = blocked (A4).
- **Config:** `PlannerConfig(connectivity=8|16, max_slope_deg, max_cross_slope_deg,
  slope_alpha)`. Defaults 25°/20°/1.0 are **placeholders** (ASSUMPTIONS A8–A9).
- **Output:** `RouteComparison` (pydantic, JSON-serializable) with three
  `RouteResult`s and a `summary`. Costs are in meters-equivalent (length ×
  dimensionless penalty); `length_3d_m` is true meters.

## Method and guarantees

- A* over a lazily-defined graph (`astar.py`), heuristic = straight-line 3-D
  Euclidean distance to goal — **admissible and consistent** for all three
  policies because every edge cost ≥ its 3-D length ≥ the straight line (A10).
- Ties broken by an insertion counter ⇒ deterministic paths (A11).
- **Validated** against NetworkX Dijkstra on the same graph over randomized
  terrain, nodata, 8/16 connectivity and all policies
  (`tests/test_astar_vs_networkx.py`): A* returns exactly the optimal cost.

## Usage

```python
from marswalk.terrain.synthetic import ridge_world
from marswalk.planner.routes import compare_routes, PlannerConfig

grid = ridge_world(40, 40, cellsize=2.0, ridge_height_m=2.0,
                   ridge_half_width=1, gap_col=20)
cmp = compare_routes(grid, start=(0, 0), goal=(39, 0),
                     config=PlannerConfig(connectivity=8, max_slope_deg=25.0))
print(cmp.summary)
for r in cmp.routes:
    print(r.name, r.mission_feasible, r.reason)
```

## Not here yet (gated by the context file)

Mission constraints (walk-back, consumables, thermal, comms) and multi-objective
NAMOA*/orienteering are **not** in the baseline. Per `MARSWALK_CONTEXT.md` §3,
multi-objective search does not start until the walk-back and consumables checks
(P4) pass tests. The slope penalty is a placeholder, not validated locomotion
energy (P4/C5 owns that).

## Known limitations

- 16-connectivity knight moves are single edges; cells they pass over are not
  checked for blockage (A5).
- Cross-slope uses the mean of endpoint gradients; cells next to nodata get NaN
  gradient and are excluded (conservative, A4).
- Thresholds are unsourced placeholders — replace with cited suit/rover values.
