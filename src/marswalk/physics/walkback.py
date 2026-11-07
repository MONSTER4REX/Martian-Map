"""Walk-back feasibility: can the explorer always get to a safe haven in time?

Rule (MARSWALK_CONTEXT.md §3): at every node and time on the plan,
``time_to_nearest_safe_haven <= remaining_consumables_time - reserve``, for the
nominal case and contingencies (rover loss, suit degradation).

Implementation:
* ``time_to_haven_field`` runs a multi-source Dijkstra from the haven set over
  the traversable graph (same HARD exclusions as the full-constraint planner),
  with edge weights = locomotion traverse time. It gives, for every cell, the
  minimum time to reach some haven on foot.
* ``check_walkback`` walks an outbound route, depleting the consumable time
  budget, and checks the rule at each node under a scenario.

All parameters are placeholders (see locomotion/consumables modules). Walk-back
is the GATE: multi-objective search must not start until this passes tests.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from pydantic import BaseModel, Field

from marswalk.physics.consumables import ConsumablesModel
from marswalk.physics.locomotion import LocomotionParams, activity_factor, edge_time_s
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.cost import horizontal_distance
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig, _edge_cost
from marswalk.planner.slope import edge_along_slope_deg, edge_cross_slope_deg, terrain_gradient


class WalkbackScenario(BaseModel):
    """A contingency. ``speed_scale`` < 1 slows the crew; ``consumption_scale``
    > 1 depletes consumables faster. Both default to nominal (1.0)."""

    name: str
    havens: list[Cell]
    speed_scale: float = Field(default=1.0, gt=0)
    consumption_scale: float = Field(default=1.0, ge=1.0)


class WalkbackReport(BaseModel):
    scenario: str
    feasible: bool
    time_budget_s: float
    reserve_s: float
    activity_factor: float
    min_margin_s: float  # smallest (remaining - reserve - time_to_haven) along route
    first_violation_index: int | None
    first_violation_cell: Cell | None
    n_nodes: int
    reason: str


def _allowed_time_edges(grid: Grid, pcfg: PlannerConfig, loco: LocomotionParams, speed_scale: float):
    """Yield (a, b, traverse_time_s) for every edge allowed under HARD exclusions."""
    g_col, g_row = terrain_gradient(grid)
    offsets = neighbor_offsets(pcfg.connectivity)
    cs = grid.cellsize
    elev = grid.elevation
    for r in range(grid.nrows):
        for c in range(grid.ncols):
            za = elev[r, c]
            if not math.isfinite(za):
                continue
            a = (r, c)
            for dr, dc in offsets:
                b = (r + dr, c + dc)
                if not grid.in_bounds(b):
                    continue
                zb = elev[b]
                if not math.isfinite(zb):
                    continue
                horiz = horizontal_distance(dr, dc, cs)
                dz = float(zb - za)
                along = edge_along_slope_deg(dz, horiz)
                cross = edge_cross_slope_deg(g_col, g_row, a, b)
                if _edge_cost("full_constraint", dz, horiz, along, cross, pcfg) is None:
                    continue
                yield a, b, edge_time_s(dz, horiz, loco) / speed_scale


def time_to_haven_field(
    grid: Grid,
    havens: list[Cell],
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    speed_scale: float = 1.0,
) -> np.ndarray:
    """Min on-foot time (s) from each cell to the nearest haven. inf = no path.

    Haven cells are 0; unknown/excluded cells and cells with no feasible path
    to any haven are ``inf``.
    """
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()

    # Reverse adjacency: to reach `a` we came from `b` paying time(a->b).
    rev: dict[Cell, list[tuple[Cell, float]]] = {}
    for a, b, t in _allowed_time_edges(grid, pcfg, loco, speed_scale):
        rev.setdefault(b, []).append((a, t))

    dist = np.full(grid.shape, math.inf, dtype=float)
    heap: list[tuple[float, int, Cell]] = []
    counter = 0
    for h in havens:
        if grid.is_known(h) and dist[h] != 0.0:
            dist[h] = 0.0
            heap.append((0.0, counter, h))
            counter += 1
    heapq.heapify(heap)

    while heap:
        d, _, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for prev, t in rev.get(node, ()):
            nd = d + t
            if nd < dist[prev]:
                dist[prev] = nd
                heapq.heappush(heap, (nd, counter, prev))
                counter += 1
    return dist


def _route_activity_factor(grid: Grid, path: list[Cell], loco: LocomotionParams) -> float:
    """Mean activity uplift over the route's edges (>= 1)."""
    if len(path) < 2:
        return 1.0
    cs = grid.cellsize
    elev = grid.elevation
    total = 0.0
    for a, b in zip(path[:-1], path[1:], strict=True):
        horiz = horizontal_distance(b[0] - a[0], b[1] - a[1], cs)
        dz = float(elev[b] - elev[a])
        total += activity_factor(edge_along_slope_deg(dz, horiz), loco)
    return total / (len(path) - 1)


def check_walkback(
    grid: Grid,
    path: list[Cell],
    scenario: WalkbackScenario,
    consumables: ConsumablesModel,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor_override: float | None = None,
) -> WalkbackReport:
    """Check the walk-back rule at every node of an outbound route.

    The budget is the consumables' binding-resource time (scaled by the
    scenario's ``consumption_scale`` and the route activity factor). Outbound
    elapsed time and the haven-time field both use the scenario's speed scale.
    """
    if not path:
        raise ValueError("path must be non-empty")
    bad = [cell for cell in path if not grid.is_known(cell)]
    if bad:
        raise ValueError(
            f"path crosses unknown/out-of-bounds cells {bad[:3]}"
            f"{'...' if len(bad) > 3 else ''}; walk-back needs a traversable route"
        )
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()

    af = (
        activity_factor_override
        if activity_factor_override is not None
        else _route_activity_factor(grid, path, loco)
    )
    budget = consumables.time_budget_s(af, scenario.consumption_scale)
    field = time_to_haven_field(grid, scenario.havens, pcfg, loco, scenario.speed_scale)

    cs = grid.cellsize
    elev = grid.elevation
    elapsed = 0.0
    min_margin = math.inf
    first_idx: int | None = None
    first_cell: Cell | None = None

    for i, node in enumerate(path):
        if i > 0:
            a = path[i - 1]
            horiz = horizontal_distance(node[0] - a[0], node[1] - a[1], cs)
            dz = float(elev[node] - elev[a])
            elapsed += edge_time_s(dz, horiz, loco) / scenario.speed_scale
        tth = float(field[node])
        margin = (budget - elapsed) - reserve_s - tth
        if margin < min_margin:
            min_margin = margin
        if margin < 0 and first_idx is None:
            first_idx = i
            first_cell = node

    feasible = first_idx is None and math.isfinite(min_margin)
    if feasible:
        reason = (
            f"OK: min walk-back margin {min_margin:.0f} s over {len(path)} nodes "
            f"(budget {budget:.0f} s, reserve {reserve_s:.0f} s)"
        )
    elif first_cell is not None and not math.isfinite(float(field[first_cell])):
        reason = (
            f"VIOLATION at node {first_idx} {first_cell}: no feasible walk-back path "
            f"to any haven ({scenario.name})"
        )
    else:
        reason = (
            f"VIOLATION at node {first_idx} {first_cell}: walk-back time exceeds "
            f"remaining consumables minus reserve ({scenario.name}); "
            f"min margin {min_margin:.0f} s"
        )

    return WalkbackReport(
        scenario=scenario.name,
        feasible=feasible,
        time_budget_s=budget,
        reserve_s=reserve_s,
        activity_factor=af,
        min_margin_s=min_margin,
        first_violation_index=first_idx,
        first_violation_cell=first_cell,
        n_nodes=len(path),
        reason=reason,
    )


def standard_scenarios(base: Cell, rover: Cell | None = None) -> list[WalkbackScenario]:
    """Nominal, rover-loss, and suit-degraded scenarios. Scales are PLACEHOLDERS."""
    havens_with_rover = [base] + ([rover] if rover is not None else [])
    return [
        WalkbackScenario(name="nominal", havens=havens_with_rover),
        WalkbackScenario(name="rover_loss", havens=[base]),
        WalkbackScenario(
            name="suit_degraded",
            havens=havens_with_rover,
            speed_scale=0.7,
            consumption_scale=1.4,
        ),
    ]
