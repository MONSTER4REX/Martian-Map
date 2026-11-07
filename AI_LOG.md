# AI authorship log (for submission disclosure)

Kept so the team can disclose AI-assisted work accurately (Open item A1 —
event rules not yet confirmed). Each row: what the AI produced, who reviewed /
owns it, and what remains human-verified. Owners: update "Reviewed by" after
you read and run the code.

> Convention: the AI (Claude, Opus 4.8) writes code, tests and docs offline on
> synthetic fixtures. Owners run on real data, review, and own their module for
> judging Q&A. Nothing network/GPU-dependent is "done" until an owner pastes
> back real results.

| Date | Author | Artifact | Reviewed by | Notes |
|---|---|---|---|---|
| 2026-11-08 | Claude (Opus 4.8) | Repo skeleton (pyproject, ruff, pytest, CI, licenses, docs) | _pending_ | — |
| 2026-11-08 | Claude (Opus 4.8) | P3 planner baseline: `grid`, `connectivity`, `slope`, `cost`, `astar`, `layers`, `routes`; synthetic terrain | _pending (P3)_ | Offline, synthetic terrain only. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | Test suite: A* vs NetworkX oracle (randomized), grid, slope, routes | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | P4 EVA physics: `locomotion`, `consumables`, `walkback` (nominal / rover-loss / suit-degraded) | _pending (P4)_ | Offline, synthetic terrain. ALL parameters are placeholders. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | P4 tests (locomotion, consumables, walk-back incl. rover-loss decision flip) | _pending (P4/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/mission.py`: walk-back wired as a hard planner constraint (time-objective A* + prune); route-level scenario flip | _pending (P3/P4)_ | Offline. Satisfies §3 gate. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | Mission tests (soundness self-check, scenario flip, infeasible goal/start) + docs/mission.md | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/multiobjective.py`: exact Pareto (NAMOA*-style Martins) on a corridor + `corridor_mask`; docs/multiobjective.md | _pending (P3)_ | Offline. Extremes validated vs NetworkX. |
| 2026-11-08 | Claude (Opus 4.8) | Multi-objective tests (dominance, non-dominated front, per-objective extremes vs Dijkstra, corridor) | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/orienteering.py`: time-windowed orienteering (exact Held-Karp subset DP) under walk-back; docs/orienteering.md | _pending (P3)_ | Offline. |
| 2026-11-08 | Claude (Opus 4.8) | Orienteering tests (budget subset, time window, walk-back/unreachable exclusion, determinism) | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | P1 `data/derivatives.py` (slope/aspect/hillshade/curvature) + `data/qa.py` (C1 DTM QA) + tests + docs/data.md | _pending (P1)_ | Offline on synthetic DEMs. |
| 2026-11-08 | Claude (Opus 4.8) | P1 `data/error_model.py` (C2 correlated-noise ensembles, empirical variogram, exponential fit) + tests | _pending (P1/P6)_ | Offline. Fit params must come from real residuals. |
| 2026-11-08 | Claude (Opus 4.8) | P6 `validation/` metrics (jaccard, 3-D length, hazard, discrete Fréchet), route stability, ablation harness + tests + docs/validation.md | _pending (P6)_ | Offline on synthetic terrain + DEM ensembles. |

## How to keep this honest

- Add a row when the AI produces or substantially rewrites a module.
- When an owner rewrites AI code by hand, note it (author changes to that owner).
- Keep commit messages descriptive; this table plus `git log` is the disclosure.
