# Multi-objective routing (spec step 5, NAMOA*-style)

Status: **offline baseline complete.** Owner: P3. Unlocked by the walk-back gate.
Module: `src/marswalk/planner/multiobjective.py`.

## What it does

Returns the **Pareto-optimal set** of routes start→goal that trade off several
objectives (all minimized), over the full-constraint traversable graph
restricted to an optional **corridor mask** (spec: multi-objective search on a
corridor only).

- `plan_pareto(grid, start, goal, objectives=("distance","time","hazard"),
  pcfg, loco, corridor)` → `ParetoFront` of `ParetoRoute(cost, path)`.
- `corridor_mask(grid, start, goal, half_width_cells)` → boolean mask around the
  start–goal segment, to keep the search tractable.
- Objectives: `distance` (3-D m), `time` (s, from locomotion), `hazard`
  (slope-penalized excess length). Any subset is selectable.

## Method and guarantee

Multi-objective label-setting (Martins' algorithm = NAMOA* with a zero
heuristic). All edge objectives are non-negative and additive, so the algorithm
returns the **exact** Pareto front. Labels are kept per node as a non-dominated
set; dominated labels are pruned; a label already dominated by a goal label is
not extended. An admissible *vector* heuristic can be added later to prune
faster without changing the result.

**Validated** (tests): the distance-minimizing and time-minimizing Pareto labels
exactly equal single-objective Dijkstra on the same graph (NetworkX oracle);
the returned front is mutually non-dominated; corridor-restricted routes stay
inside the mask.

## Caveats

- Front size grows with objectives/terrain; always restrict to a corridor on
  large AOIs (the search raises if it exceeds `max_expansions`).
- `hazard`/`time` use the placeholder slope penalty and locomotion model
  (P4 to replace with cited values).
- Walk-back-constrained multi-objective (Pareto over routes that are also
  walk-back-safe) = combine this with `mission.py`'s prune; a later step.
