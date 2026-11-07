"""Walk-back as a hard mission constraint on the route (spec step 4).

The full-constraint planner (`routes.py`) excludes terrain you must not cross
(slope, cross-slope, unknown). This adds the *mission* constraint: the route
must keep a feasible walk-back to a safe haven at every point, under a chosen
contingency (nominal / rover-loss / suit-degraded).

Formulation (sound and complete for the minimum-time walk-back-safe route):

* Search minimizes **outbound traverse time** (walk-back is time-based), using
  the same hard terrain exclusions as `full_constraint`.
* A precomputed haven-time field ``tth`` gives the on-foot time from any cell
  back to the nearest haven.
* An edge into cell ``v`` is allowed only if, at the arrival time ``t(v)`` along
  the path being built, ``t(v) + tth(v) + reserve <= budget``.

Because elapsed time only grows along a path and ``tth`` is fixed per cell, the
earliest (minimum-time) arrival at ``v`` is the one most likely to satisfy the
rule; if it fails, every later arrival fails too, so pruning on it is safe and
loses no feasible route. A* on time with a consistent heuristic therefore
returns the optimal (min-time) route that is walk-back-safe at every node.

NOTE: the objective here is time, distinct from the slope-distance mission route
in `routes.py`. Trading those objectives off (distance vs time vs walk-back
margin) is the step-5 NAMOA* job, which this gate unlocks.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from pydantic import BaseModel

from marswalk.physics.consumables import ConsumablesModel
from marswalk.physics.locomotion import LocomotionParams, edge_time_s
from marswalk.physics.walkback import (
    WalkbackScenario,
    check_walkback,
    time_to_haven_field,
)
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.cost import horizontal_distance
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import (
    PlannerConfig,
    RouteComparison,
    _edge_cost,
    compare_routes,
)
from marswalk.planner.slope import edge_along_slope_deg, edge_cross_slope_deg, terrain_gradient


class WalkbackRouteResult(BaseModel):
    scenario: str
    feasible: bool
    reason: str
    path: list[Cell] | None
    total_time_s: float | None
    budget_s: float
    reserve_s: float
    min_margin_s: float | None  # verified via check_walkback on the found path


class MissionComparison(BaseModel):
    """The three distance routes plus the walk-back-safe route and the flip."""

    base: RouteComparison
    scenario: str
    walkback_route: WalkbackRouteResult
    shortest_feasible_walkback: bool  # is the distance-shortest route walk-back-safe?
    shortest_walkback_reason: str
    summary: str


def _reconstruct(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plan_walkback_route(
    grid: Grid,
    start: Cell,
    goal: Cell,
    scenario: WalkbackScenario,
    consumables: ConsumablesModel,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor: float = 1.0,
) -> WalkbackRouteResult:
    """Minimum-time route that keeps a feasible walk-back at every node."""
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()
    budget = consumables.time_budget_s(activity_factor, scenario.consumption_scale)

    def _fail(reason: str) -> WalkbackRouteResult:
        return WalkbackRouteResult(
            scenario=scenario.name,
            feasible=False,
            reason=reason,
            path=None,
            total_time_s=None,
            budget_s=budget,
            reserve_s=reserve_s,
            min_margin_s=None,
        )

    for label, cell in (("start", start), ("goal", goal)):
        if not grid.in_bounds(cell):
            return _fail(f"{label} cell {cell} out of bounds {grid.shape}")
        if not grid.is_known(cell):
            return _fail(f"{label} cell {cell} is unknown/nodata")

    g_col, g_row = terrain_gradient(grid)
    tth = time_to_haven_field(grid, scenario.havens, pcfg, loco, scenario.speed_scale)

    if not math.isfinite(tth[start]):
        return _fail(f"start has no feasible walk-back to any haven ({scenario.name})")
    if tth[start] + reserve_s > budget:
        return _fail(
            f"infeasible at start: walk-back {tth[start]:.0f} s + reserve "
            f"{reserve_s:.0f} s exceeds budget {budget:.0f} s ({scenario.name})"
        )

    path, total_time = _astar_time(
        grid, g_col, g_row, start, goal, pcfg, loco, scenario, tth, budget, reserve_s
    )
    if path is None:
        return _fail(
            f"no walk-back-safe route to {goal}: goal unreachable within budget "
            f"{budget:.0f} s / reserve {reserve_s:.0f} s under {scenario.name}"
        )

    # Independent verification on the found path (soundness self-check).
    report = check_walkback(
        grid, path, scenario, consumables, pcfg, loco, reserve_s, activity_factor
    )
    return WalkbackRouteResult(
        scenario=scenario.name,
        feasible=True,
        reason=f"walk-back-safe under {scenario.name}; {report.reason}",
        path=path,
        total_time_s=total_time,
        budget_s=budget,
        reserve_s=reserve_s,
        min_margin_s=report.min_margin_s,
    )


def _astar_time(
    grid: Grid,
    g_col: np.ndarray,
    g_row: np.ndarray,
    start: Cell,
    goal: Cell,
    pcfg: PlannerConfig,
    loco: LocomotionParams,
    scenario: WalkbackScenario,
    tth: np.ndarray,
    budget: float,
    reserve_s: float,
) -> tuple[list[Cell] | None, float]:
    offsets = neighbor_offsets(pcfg.connectivity)
    cs = grid.cellsize
    elev = grid.elevation
    zg = elev[goal]
    max_speed = loco.level_speed_ms * scenario.speed_scale  # fastest possible

    def h(node: Cell) -> float:
        horiz = cs * math.hypot(node[0] - goal[0], node[1] - goal[1])
        dz = float(elev[node] - zg)
        return math.hypot(horiz, dz) / max_speed  # admissible time lower bound

    if start == goal:
        return [start], 0.0

    counter = 0
    heap: list[tuple[float, float, int, Cell]] = [(h(start), 0.0, counter, start)]
    g_time: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    closed: set[Cell] = set()

    while heap:
        _, gt, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            return _reconstruct(came_from, cur), gt
        closed.add(cur)
        r, c = cur
        za = elev[cur]
        for dr, dc in offsets:
            b = (r + dr, c + dc)
            if not grid.in_bounds(b) or b in closed:
                continue
            zb = elev[b]
            if not math.isfinite(zb):
                continue
            horiz = horizontal_distance(dr, dc, cs)
            dz = float(zb - za)
            along = edge_along_slope_deg(dz, horiz)
            cross = edge_cross_slope_deg(g_col, g_row, cur, b)
            if _edge_cost("full_constraint", dz, horiz, along, cross, pcfg) is None:
                continue
            if not math.isfinite(tth[b]):
                continue  # b has no walk-back path
            tentative = gt + edge_time_s(dz, horiz, loco) / scenario.speed_scale
            # Walk-back rule at arrival time tentative.
            if tentative + float(tth[b]) + reserve_s > budget:
                continue
            if tentative < g_time.get(b, math.inf):
                g_time[b] = tentative
                came_from[b] = cur
                counter += 1
                heapq.heappush(heap, (tentative + h(b), tentative, counter, b))

    return None, math.inf


def compare_routes_with_walkback(
    grid: Grid,
    start: Cell,
    goal: Cell,
    scenario: WalkbackScenario,
    consumables: ConsumablesModel,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor: float = 1.0,
) -> MissionComparison:
    """Distance routes + the walk-back-safe route, with the decision-flip story."""
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()

    base = compare_routes(grid, start, goal, pcfg)
    wb = plan_walkback_route(
        grid, start, goal, scenario, consumables, pcfg, loco, reserve_s, activity_factor
    )

    shortest = base.by_name("Shortest (3-D distance)")
    if shortest.feasible and shortest.path is not None:
        chk = check_walkback(
            grid, shortest.path, scenario, consumables, pcfg, loco, reserve_s, activity_factor
        )
        shortest_ok = chk.feasible
        shortest_reason = chk.reason
    else:
        shortest_ok = False
        shortest_reason = f"shortest route not feasible: {shortest.reason}"

    summary = _mission_summary(scenario.name, shortest_ok, wb)
    return MissionComparison(
        base=base,
        scenario=scenario.name,
        walkback_route=wb,
        shortest_feasible_walkback=shortest_ok,
        shortest_walkback_reason=shortest_reason,
        summary=summary,
    )


def _mission_summary(scenario: str, shortest_ok: bool, wb: WalkbackRouteResult) -> str:
    if not shortest_ok and wb.feasible:
        return (
            f"DECISION FLIP under {scenario}: the shortest route is NOT "
            f"walk-back-safe, but a walk-back-safe route exists "
            f"(min margin {wb.min_margin_s:.0f} s)."
        )
    if not shortest_ok and not wb.feasible:
        return (
            f"Under {scenario}: neither the shortest route nor any walk-back-safe "
            f"route to the goal is viable - {wb.reason}."
        )
    if shortest_ok and wb.feasible:
        return (
            f"Under {scenario}: the shortest route is already walk-back-safe; "
            f"no detour needed."
        )
    return f"Under {scenario}: shortest route walk-back-safe but planner reported {wb.reason}."
