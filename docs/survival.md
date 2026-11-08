# Survival envelope — the human core (and the hero visual)

Module: `physics/survival.py`. This turns the walk-back constraint into the
thing that actually matters to a crew: **where can I go and still get home
alive?** — and shows how that world collapses under contingencies.

## The idea

From a start point, a cell is **survivable** only if

```
time_to_reach(start → cell) + time_to_haven(cell → nearest haven) + reserve ≤ budget
```

for the active contingency. Cells that fail are **no-go / death zones**: you
could walk there, but you could not get back before consumables run out. Lose
the rover (a haven) or degrade the suit (slower + faster burn) and the
survivable world shrinks — often dramatically.

## API

- `reach_time_field(grid, start, …)` — forward Dijkstra: on-foot time from
  `start` to every cell (inf if blocked).
- `survival_envelope(grid, start, scenario, consumables, …) → SurvivalEnvelope`
  with `reach_s`, `haven_s`, `margin_s` (spare time), `survivable` mask,
  `no_go()`, and `survivable_fraction`.
- `compare_survival(grid, start, scenarios, …) → SurvivalComparison`;
  `shrink_vs_first()` gives the fraction of the safe world lost vs nominal.

Reach uses a forward Dijkstra; haven time reuses `walkback.time_to_haven_field`;
both use the full-constraint hard exclusions and the scenario's speed scale.

## Demo

`scripts/survival_demo.py` renders the three-panel hero figure
(`docs/survival_envelope.png`) — nominal vs rover-lost vs suit-degraded — on a
realistic km-scale AOI, with the budget auto-calibrated so the nominal world is
~80% survivable and you watch it collapse (e.g. **80% → 47% → 7%**).

```bash
python scripts/survival_demo.py
```

## Validity

Terrain is synthetic and **all suit/rover/consumables numbers are
PLACEHOLDERS** (see `physics/locomotion.py`, `physics/consumables.py`). The
*mechanism* is what this provides; an owner's real Jezero DEM and cited
life-support figures drop straight in and the map becomes real. The walk-back
rule and fields are the same ones tested in `docs/physics.md` /
`docs/mission.md`.
