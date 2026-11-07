"""Science under walk-back: time-windowed orienteering (spec step 6).

The user picks science targets, each with a value and an optional time window.
The planner selects and orders a subset to **maximize total science value**
while:

* starting at ``base`` and returning to ``base`` within the consumables budget
  minus a safety reserve (a closed tour), and
* keeping a feasible **walk-back** at every visited target (arrival time +
  time-to-nearest-haven + reserve <= budget) under a chosen contingency.

This only runs after the walk-back gate (it depends on it). It is NP-hard in
general; for a small, user-picked target set (default <= 15) we solve it
*exactly* with a Held-Karp-style subset DP that stores the minimum feasible
arrival time per (visited-set, last-target) and respects time windows.

Travel times are shortest on-foot times over the full-constraint traversable
graph (symmetric in magnitude under the |slope| speed model), from
`time_to_haven_field` seeded at each node.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from marswalk.physics.consumables import ConsumablesModel
from marswalk.physics.locomotion import LocomotionParams
from marswalk.physics.walkback import WalkbackScenario, time_to_haven_field
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig


class ScienceTarget(BaseModel):
    cell: Cell
    value: float = Field(ge=0)
    open_s: float = 0.0  # earliest arrival (wait if early)
    close_s: float = math.inf  # latest arrival
    service_s: float = 0.0  # time spent at the target


class OrienteeringResult(BaseModel):
    order: list[int]  # indices into the input targets, in visit order
    visited_cells: list[Cell]  # base -> targets... -> base
    total_value: float
    total_time_s: float  # closed-tour time incl. travel + service + waiting
    budget_s: float
    reserve_s: float
    feasible: bool
    n_targets: int
    excluded: list[int]  # targets never reachable/safe to visit
    reason: str


def plan_orienteering(
    grid: Grid,
    base: Cell,
    targets: list[ScienceTarget],
    consumables: ConsumablesModel,
    scenario: WalkbackScenario,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor: float = 1.0,
    max_targets: int = 15,
) -> OrienteeringResult:
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()
    n = len(targets)
    budget = consumables.time_budget_s(activity_factor, scenario.consumption_scale)

    if n > max_targets:
        raise ValueError(f"{n} targets exceeds max_targets={max_targets}; use a corridor/cluster")
    if not grid.is_known(base):
        raise ValueError(f"base {base} is unknown/out of bounds")

    # Shortest on-foot time from each node (base + targets) to every cell.
    nodes: list[Cell] = [base] + [t.cell for t in targets]
    fields = [time_to_haven_field(grid, [nd], pcfg, loco, scenario.speed_scale) for nd in nodes]

    def travel(i: int, j: int) -> float:
        return float(fields[i][nodes[j]])  # symmetric time i<->j

    # Walk-back time field for the scenario (abort-to-haven).
    tth = time_to_haven_field(grid, scenario.havens, pcfg, loco, scenario.speed_scale)

    # Targets that can never be part of a safe tour (unreachable, or too far to
    # keep walk-back even on the fastest direct arrival).
    excluded: list[int] = []
    for k, t in enumerate(targets):
        j = k + 1
        arrive = max(t.open_s, travel(0, j))
        unreachable = not math.isfinite(travel(0, j)) or not math.isfinite(float(tth[t.cell]))
        if (
            unreachable
            or arrive > t.close_s
            or arrive + float(tth[t.cell]) + reserve_s > budget
        ):
            excluded.append(k)

    usable = [k for k in range(n) if k not in excluded]

    # DP over subsets of `usable`. Bit i in mask <-> usable[i] visited.
    m = len(usable)
    # dp[mask][last_local] = (min_arrival_time, parent_mask, parent_local)
    NEG = None
    dp: list[list[tuple[float, int, int] | None]] = [[NEG] * m for _ in range(1 << m)]

    def feas_visit(local: int, arrive: float) -> bool:
        t = targets[usable[local]]
        if arrive > t.close_s:
            return False
        return arrive + float(tth[t.cell]) + reserve_s <= budget

    for li in range(m):
        t = targets[usable[li]]
        arrive = max(t.open_s, travel(0, usable[li] + 1))
        if math.isfinite(arrive) and feas_visit(li, arrive):
            dp[1 << li][li] = (arrive, 0, -1)

    for mask in range(1 << m):
        for last in range(m):
            cur = dp[mask][last]
            if cur is None or not (mask & (1 << last)):
                continue
            arr_last, _, _ = cur
            depart = arr_last + targets[usable[last]].service_s
            for nxt in range(m):
                if mask & (1 << nxt):
                    continue
                arrive = max(
                    targets[usable[nxt]].open_s,
                    depart + travel(usable[last] + 1, usable[nxt] + 1),
                )
                if not math.isfinite(arrive) or not feas_visit(nxt, arrive):
                    continue
                nmask = mask | (1 << nxt)
                prev = dp[nmask][nxt]
                if prev is None or arrive < prev[0]:
                    dp[nmask][nxt] = (arrive, mask, last)

    # Pick the subset of maximum value whose closed tour fits budget - reserve.
    best_value = 0.0
    best_time = 0.0
    best_mask = 0
    best_last = -1
    for mask in range(1 << m):
        value = sum(targets[usable[li]].value for li in range(m) if mask & (1 << li))
        for last in range(m):
            cur = dp[mask][last]
            if cur is None:
                continue
            arr_last, _, _ = cur
            completion = arr_last + targets[usable[last]].service_s + travel(usable[last] + 1, 0)
            if completion + reserve_s > budget:
                continue
            if value > best_value or (value == best_value and completion < best_time):
                best_value, best_time, best_mask, best_last = value, completion, mask, last

    order = _reconstruct_order(dp, best_mask, best_last, usable)
    visited_cells = [base] + [targets[k].cell for k in order] + [base]
    reason = (
        f"visited {len(order)} of {n} target(s) for value {best_value:g}; "
        f"tour {best_time:.0f} s + reserve {reserve_s:.0f} s <= budget {budget:.0f} s"
        if order
        else f"no target safely visitable within budget {budget:.0f} s (reserve {reserve_s:.0f} s)"
    )
    return OrienteeringResult(
        order=order,
        visited_cells=visited_cells,
        total_value=best_value,
        total_time_s=best_time,
        budget_s=budget,
        reserve_s=reserve_s,
        feasible=True,  # doing nothing is always feasible; value may be 0
        n_targets=n,
        excluded=sorted(excluded),
        reason=reason,
    )


def _reconstruct_order(
    dp: list[list[tuple[float, int, int] | None]],
    mask: int,
    last: int,
    usable: list[int],
) -> list[int]:
    if last < 0 or mask == 0:
        return []
    order_local: list[int] = []
    cur_mask, cur_last = mask, last
    while cur_last >= 0 and dp[cur_mask][cur_last] is not None:
        order_local.append(cur_last)
        _, pmask, plast = dp[cur_mask][cur_last]  # type: ignore[misc]
        cur_mask, cur_last = pmask, plast
    order_local.reverse()
    return [usable[li] for li in order_local]
