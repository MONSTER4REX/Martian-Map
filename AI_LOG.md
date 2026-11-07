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

## How to keep this honest

- Add a row when the AI produces or substantially rewrites a module.
- When an owner rewrites AI code by hand, note it (author changes to that owner).
- Keep commit messages descriptive; this table plus `git log` is the disclosure.
