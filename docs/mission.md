# Mission constraint: walk-back wired into the planner (spec step 4)

Status: **offline baseline complete, UNTESTED ON REAL DATA.** Owners: P3 + P4.
This completes the §3 **gate** — walk-back is now a hard planner constraint, not
just a standalone check — so step 5 (multi-objective NAMOA*) is unlocked.

Module: `src/marswalk/planner/mission.py`.

## What it adds

`routes.py` excludes terrain you must not cross (slope, cross-slope, unknown).
`mission.py` adds the contingency constraint: the route must keep a feasible
**walk-back** to a safe haven at every point.

- `plan_walkback_route(grid, start, goal, scenario, consumables, …, reserve_s)`
  → `WalkbackRouteResult` (feasible / path / total_time_s / min_margin_s /
  reason). Minimum-time route that is walk-back-safe at every node.
- `compare_routes_with_walkback(…)` → `MissionComparison`: the three distance
  routes, the walk-back-safe route, whether the shortest distance route is
  itself walk-back-safe, and a plain summary of the decision.

## Why time (not slope-distance) is the search objective here

Walk-back is a time/consumables statement. The constraint at a cell `v` reached
at elapsed time `t(v)` is:

```
t(v) + time_to_haven(v) + reserve  <=  consumables_time_budget
```

Searching on **outbound traverse time** makes `t(v)` the quantity the constraint
needs, and lets a single haven-time field (`time_to_haven_field`) answer the
walk-back term for every cell. The slope-distance mission route in `routes.py`
stays as a separate objective; trading the two off is step-5 NAMOA* work.

## Soundness of the prune

An edge into `v` is kept only if `t(v) + tth(v) + reserve <= budget`, evaluated
at the arrival time of the path being built. Elapsed time only grows along a
path and `tth` is fixed per cell, so the earliest (min-time) arrival at `v` is
the most permissive; if it fails, every later arrival fails too. Pruning on it
therefore removes no feasible route, and A* on time with a consistent heuristic
(`straight-line 3-D distance / max speed`) returns the optimal min-time
walk-back-safe route. Each found route is independently re-verified with
`check_walkback` (soundness self-check) before it is returned.

## Decision flip (spec demo a), scenario form

Flat 40-cell corridor, far goal, base at the near end, rover at the far end,
tight consumable budget:

| Scenario | Shortest route walk-back-safe? | Any walk-back-safe route? | Decision |
|---|---|---|---|
| nominal (rover is a haven) | ✅ | ✅ | **go** |
| rover_loss (base only) | ❌ | ❌ (goal genuinely too far) | **no-go** |

Losing the rover removes a haven, pushing the far half of the trip past the
walk-back limit — the mission decision flips from "go" to "no safe route". This
is the defensible, physical flip on these models. A *longer-detour* flip (a
walk-back-safe route that trades distance for margin) is a multi-objective
trade-off and belongs to step 5, now unlocked.

## Caveats

- All P4 parameters remain placeholders (see `docs/physics.md`).
- Objective is min-time; distance/energy/time trade-offs are step 5.
- Downhill timing equals uphill (|slope|) — conservative placeholder.
