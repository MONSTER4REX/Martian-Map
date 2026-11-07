# Science under walk-back: time-windowed orienteering (spec step 6)

Status: **offline baseline complete.** Owner: P3. Depends on the walk-back gate.
Module: `src/marswalk/planner/orienteering.py`.

## What it does

The user picks science `ScienceTarget`s (cell, value, optional `[open_s,
close_s]` window, optional `service_s`). The planner selects and orders a
subset to **maximize total science value** subject to:

- a closed tour `base → … → base` within `budget − reserve` (consumables), and
- a feasible **walk-back** at every visited target: `arrival + time_to_haven +
  reserve ≤ budget`, under a chosen `WalkbackScenario`.

Returns `OrienteeringResult(order, visited_cells, total_value, total_time_s,
excluded, reason, …)`.

## Method

- Travel times = shortest on-foot time over the full-constraint traversable
  graph, from `time_to_haven_field` seeded at each node (symmetric under the
  |slope| speed model).
- Targets that are unreachable, past their window on the fastest direct
  arrival, or beyond the walk-back limit are pre-`excluded`.
- Exact **Held-Karp subset DP**: `dp[visited_set][last]` = min feasible arrival
  time, respecting time windows and the per-target walk-back rule. The best
  subset is the max-value one whose closed tour fits `budget − reserve`.
- NP-hard in general, so capped at `max_targets` (default 15); raise for more
  and cluster/corridor first.

## Tests

Generous budget visits all; a tight budget picks the higher-value side; a
too-early window excludes a target; a too-far target is excluded by walk-back; a
nodata-walled target is excluded as unreachable; results are deterministic.

## Caveats

- Values/windows are user inputs; the science-value raster (C9) will feed them
  later. Budget/reserve/locomotion are placeholders (P4).
- Walk-back here is the abort-to-haven check at targets plus the planned return;
  it does not yet verify walk-back along every intermediate cell of each leg
  (the `mission.py` per-cell prune does that for a single route) — combining the
  two is a later refinement.
