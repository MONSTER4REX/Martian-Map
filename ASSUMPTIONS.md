# Assumptions and decisions log

Every entry is a decision we made so we could proceed, with its rationale and
how to revisit it. Baseline-planner assumptions are marked **[P3]**. Anything
touching real data is **UNTESTED ON REAL DATA** until an owner confirms.

## Project-level (from the three blocking questions, defaults assumed)

- **A0 Repo state:** no pre-existing repo. This skeleton is the starting point.
- **A1 Event rules UNCONFIRMED:** pre-event work and AI-use disclosure not yet
  checked against the 2026 Participant Terms. Until confirmed, all code is
  treated as prep, AI authorship is logged in `AI_LOG.md`, and AI use will be
  disclosed in the submission. (Open item — lead to confirm.)
- **A2 AOI = Jezero**, roles P1–P6 and machine assignments exactly as in
  `MARSWALK_CONTEXT.md` §3–4. Not yet re-confirmed against a coverage audit.

## Planner baseline [P3]

- **A3 Grid model:** the AOI is a single 2.5-D elevation grid (one height per
  cell), square cells of `cellsize` meters. Overhangs/caves are out of scope.
- **A4 Missing data = blocked.** A cell whose elevation is NaN (nodata) is
  *unknown* and is never traversed, in every route policy. Cells adjacent to
  nodata get NaN terrain gradient and are therefore excluded by the
  cross-slope rule as well (conservative).
- **A5 Connectivity:** 8- or 16-neighbor lattice. 16 adds the knight-style
  offsets (±1,±2),(±2,±1) to cut metrication error. A 16-move is treated as a
  single straight edge between its endpoints (it is **not** checked for
  blocked cells it passes over — revisit if this matters).
- **A6 Edge cost = true 3-D segment length** `sqrt(horizontal^2 + dz^2)`,
  where `horizontal = cellsize * sqrt(dr^2 + dc^2)` and `dz` is the elevation
  difference. This is the "shortest" objective.
- **A7 Slope-aware cost** = 3-D length × a **monotone penalty ≥ 1** that grows
  with along-path slope. The penalty is a **placeholder**, not validated
  locomotion energy — P4 (C5) owns real energy/consumables. Because the
  penalty is ≥ 1, the straight-line 3-D distance stays an admissible A*
  heuristic for all three policies.
- **A8 Along-path slope** of an edge = `atan2(|dz|, horizontal)`. Hard
  exclusion when it exceeds `max_slope_deg` (default 25°, a planning
  placeholder — revisit per suit/rover spec).
- **A9 Cross-slope** of an edge = the terrain-gradient component perpendicular
  to travel, from the mean of the two endpoints' gradients, as an angle. Hard
  exclusion above `max_cross_slope_deg` (default 20°, placeholder). Terrain
  gradient uses central differences (`numpy.gradient`) scaled by `cellsize`.
- **A10 Heuristic admissibility:** A* uses straight-line 3-D Euclidean distance
  to the goal. Valid because every edge cost ≥ its 3-D length ≥ the straight
  line (triangle inequality). If a future cost can dip below 3-D length, the
  heuristic must be rescaled or the search becomes non-optimal.
- **A11 Determinism:** A* breaks ties with an insertion counter; synthetic
  terrain and all tests use fixed seeds, so routes are reproducible.

## Layer classification (baseline)

| Layer | Class | Note |
|---|---|---|
| elevation (DEM) | informational (base) | source of all derivatives |
| nodata / unknown mask | **hard** | always blocks (A4) |
| along-path slope | **hard** + objective | exclusion (A8) and cost (A7) |
| cross-slope | **hard** | exclusion (A9) |

Mission constraints (walk-back, consumables, thermal, comms) are **not** in the
baseline — they arrive with P4/P1 and must pass their own tests first
(`MARSWALK_CONTEXT.md` §3 planner order: do not start multi-objective search
until walk-back and consumables checks pass).

## Thresholds that are placeholders (need an owner + source)

`max_slope_deg=25`, `max_cross_slope_deg=20`, slope penalty shape/`alpha`.
These are plausible planning defaults, **not** sourced from a suit/rover spec.
P4 to replace with cited values and a validity range.
