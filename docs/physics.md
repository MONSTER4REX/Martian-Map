# P4 EVA physics: locomotion, consumables, walk-back

Status: **offline baseline complete, UNTESTED ON REAL DATA.** Owner: P4.
This module is the **gate** for multi-objective search (MARSWALK_CONTEXT.md §3):
walk-back and consumables must pass tests before NAMOA*/orienteering starts.
That gate is now met in the baseline sense (tests pass on synthetic terrain).

## ⚠ Parameters are placeholders

Every number below is a **round, documented stand-in**, not a sourced value.
Mars surface gravity (3.71 m/s²) is the only cited constant. P4 must replace the
speed curve, energy coefficients, consumable capacities/rates, reserve and the
contingency scales with **cited suit/rover values and a stated validity range**.

## Modules, inputs, outputs, units

### `locomotion.py`
- `LocomotionParams` — speed curve, mass, cost-of-transport, muscle efficiency.
- `speed_ms(along_slope_deg, p)` → m/s. Monotone decreasing in |slope|, floored.
- `edge_time_s(dz, horizontal, p)` → s. `hypot(dz,horizontal)/speed`.
- `activity_factor(along_slope_deg, p)` → ≥1 metabolic/consumable uplift.
- `edge_energy_j(dz, horizontal, p)` → J. Level cost-of-transport + climb work
  (no descent credit; conservative).
- Simplification: downhill treated like uphill (|slope|) for timing.

### `consumables.py`
- `Consumable(name, capacity, rate_per_s, unit)`; `duration_s(activity)`.
- `ConsumablesModel(items)` — `time_budget_s(activity, degradation)` = the
  **binding (shortest-lasting) resource**; `binding_resource(...)` names it.
- `BASELINE_CONSUMABLES`, `CONSERVATIVE_CONSUMABLES` — placeholder sets. Units
  are abstract seconds until real life-support figures land.

### `walkback.py`
- `time_to_haven_field(grid, havens, pcfg, loco, speed_scale)` → array of min
  on-foot time (s) to the nearest haven, via **multi-source Dijkstra** over the
  graph with the **same hard exclusions as the full-constraint planner**
  (unknown / slope / cross-slope). `inf` = no feasible walk-back.
- `WalkbackScenario(name, havens, speed_scale, consumption_scale)`.
- `standard_scenarios(base, rover)` → `nominal`, `rover_loss`, `suit_degraded`.
- `check_walkback(grid, path, scenario, consumables, …, reserve_s)` →
  `WalkbackReport`. Enforces, at every node:
  `time_to_haven ≤ (budget − elapsed) − reserve`. Rejects a path that crosses
  unknown cells.

## The walk-back rule

At every node and time on the outbound route, under each scenario:

```
time_to_nearest_haven(node)  ≤  (time_budget − elapsed_outbound)  −  reserve
```

`time_budget` = binding consumable duration × (activity, degradation scales).
A single violated node makes the route infeasible for that scenario, and the
report names the first violating node.

## Demo: decision flip (spec demo a)

Flat 60-cell corridor, base at one end, rover at the far end, tight budget:

| Scenario | Feasible | Budget | Min margin | First violation |
|---|---|---|---|---|
| nominal | ✅ | 200 s | +62 s | — |
| rover_loss | ❌ | 200 s | −56 s | col 46 |
| suit_degraded | ❌ | 143 s | −46 s | col 22 |

The shortest/nominal plan is safe **only while the rover is a haven**; losing it
makes the far half of the route a walk-back violation. This is the input the
planner's mission-constraint layer will consume next.

## Next

Wire walk-back into the planner as a hard mission constraint (so the
full-constraint route *respects* walk-back under a chosen contingency), then —
and only then — begin multi-objective NAMOA* on a corridor (gate satisfied).
