"""Three-route comparison: shortest, slope-aware, full-constraint.

One destination, three policies (MARSWALK_CONTEXT.md §3 item 2):

* ``shortest``        - minimize 3-D path length; only unknown cells blocked.
* ``slope_aware``     - minimize slope-penalized length; only unknown blocked.
* ``full_constraint`` - slope-penalized length under HARD exclusions
                        (unknown cells, along-slope, cross-slope).

Every result carries its metrics, whether it is mission-feasible (satisfies the
full-constraint hard rules), and a plain-language reason when it is rejected.
This is the baseline for the decision-flip demos; mission constraints
(walk-back, consumables, thermal, comms) are NOT here yet — they arrive with
P4/P1 and must pass their own tests first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
from pydantic import BaseModel, Field

from marswalk.planner.astar import astar
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.cost import distance_3d, horizontal_distance, slope_multiplier
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.slope import edge_along_slope_deg, edge_cross_slope_deg, terrain_gradient

Policy = str
POLICIES: tuple[Policy, ...] = ("shortest", "slope_aware", "full_constraint")


class PlannerConfig(BaseModel):
    """Thresholds and policy knobs. Defaults are PLACEHOLDERS (see ASSUMPTIONS)."""

    connectivity: int = Field(default=8)
    max_slope_deg: float = Field(default=25.0, gt=0)
    max_cross_slope_deg: float = Field(default=20.0, gt=0)
    slope_alpha: float = Field(default=1.0, ge=0)

    def model_post_init(self, _ctx) -> None:
        if self.connectivity not in (8, 16):
            raise ValueError(f"connectivity must be 8 or 16, got {self.connectivity}")


class RouteMetrics(BaseModel):
    total_cost: float
    length_3d_m: float
    n_cells: int
    max_along_slope_deg: float
    mean_along_slope_deg: float
    max_cross_slope_deg: float  # max over finite edges; NaN if none finite
    n_cross_slope_undefined: int  # edges whose cross-slope is undefined (near nodata)
    n_hard_violations: int  # edges that break the full-constraint hard rules


class RouteResult(BaseModel):
    name: str
    policy: Policy
    feasible: bool  # a path exists under this policy
    mission_feasible: bool  # the path also satisfies full-constraint hard rules
    reason: str
    path: list[Cell] | None
    metrics: RouteMetrics | None


class RouteComparison(BaseModel):
    start: Cell
    goal: Cell
    config: PlannerConfig
    routes: list[RouteResult]
    summary: str

    def by_name(self, name: str) -> RouteResult:
        for r in self.routes:
            if r.name == name:
                return r
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Graph callbacks
# --------------------------------------------------------------------------- #
def _edge_cost(
    policy: Policy, dz: float, horiz: float, along: float, cross: float, cfg: PlannerConfig
) -> float | None:
    """Edge cost for a policy, or None if the edge is a hard exclusion."""
    dist = distance_3d(dz, horiz)
    if policy == "shortest":
        return dist
    base = dist * slope_multiplier(along, cfg.slope_alpha)
    if policy == "slope_aware":
        return base
    # full_constraint: apply hard exclusions
    if along > cfg.max_slope_deg:
        return None
    if not math.isfinite(cross) or cross > cfg.max_cross_slope_deg:
        return None
    return base


def _make_neighbors(
    grid: Grid, g_col: np.ndarray, g_row: np.ndarray, cfg: PlannerConfig, policy: Policy
) -> Callable[[Cell], Iterable[tuple[Cell, float]]]:
    offsets = neighbor_offsets(cfg.connectivity)
    cs = grid.cellsize
    elev = grid.elevation
    need_cross = policy == "full_constraint"

    def neighbors(node: Cell) -> Iterable[tuple[Cell, float]]:
        r, c = node
        za = elev[r, c]
        for dr, dc in offsets:
            b = (r + dr, c + dc)
            if not grid.in_bounds(b):
                continue
            zb = elev[b]
            if not math.isfinite(zb):  # unknown cell: always blocked (A4)
                continue
            horiz = horizontal_distance(dr, dc, cs)
            dz = float(zb - za)
            along = edge_along_slope_deg(dz, horiz)
            cross = edge_cross_slope_deg(g_col, g_row, node, b) if need_cross else math.nan
            cost = _edge_cost(policy, dz, horiz, along, cross, cfg)
            if cost is None:
                continue
            yield b, cost

    return neighbors


def _make_heuristic(grid: Grid, goal: Cell) -> Callable[[Cell], float]:
    gr, gc = goal
    zg = grid.elevation[goal]
    cs = grid.cellsize
    elev = grid.elevation

    def h(node: Cell) -> float:
        r, c = node
        horiz = cs * math.hypot(r - gr, c - gc)
        dz = float(elev[r, c] - zg)
        return math.hypot(horiz, dz)

    return h


# --------------------------------------------------------------------------- #
# Planning + metrics
# --------------------------------------------------------------------------- #
def plan_route(
    grid: Grid, start: Cell, goal: Cell, policy: Policy, config: PlannerConfig | None = None
) -> tuple[list[Cell] | None, float]:
    """Plan a single policy's route. Returns (path, total_cost)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    cfg = config or PlannerConfig()
    g_col, g_row = terrain_gradient(grid)
    neighbors = _make_neighbors(grid, g_col, g_row, cfg, policy)
    heuristic = _make_heuristic(grid, goal)
    return astar(start, goal, neighbors, heuristic)


def _edges(path: list[Cell]) -> list[tuple[Cell, Cell]]:
    return list(zip(path[:-1], path[1:], strict=True))


def _evaluate_path(
    grid: Grid, g_col: np.ndarray, g_row: np.ndarray, path: list[Cell], cfg: PlannerConfig
) -> tuple[RouteMetrics, bool]:
    """Metrics for a path + whether it satisfies full-constraint hard rules."""
    cs = grid.cellsize
    elev = grid.elevation
    length = 0.0
    cost = 0.0
    alongs: list[float] = []
    crosses_finite: list[float] = []
    n_undef = 0
    n_viol = 0
    for a, b in _edges(path):
        dr, dc = b[0] - a[0], b[1] - a[1]
        horiz = horizontal_distance(dr, dc, cs)
        dz = float(elev[b] - elev[a])
        dist = distance_3d(dz, horiz)
        along = edge_along_slope_deg(dz, horiz)
        cross = edge_cross_slope_deg(g_col, g_row, a, b)
        length += dist
        cost += dist * slope_multiplier(along, cfg.slope_alpha)
        alongs.append(along)
        violated = along > cfg.max_slope_deg
        if math.isfinite(cross):
            crosses_finite.append(cross)
            if cross > cfg.max_cross_slope_deg:
                violated = True
        else:
            n_undef += 1
            violated = True
        if violated:
            n_viol += 1

    metrics = RouteMetrics(
        total_cost=cost,
        length_3d_m=length,
        n_cells=len(path),
        max_along_slope_deg=max(alongs) if alongs else 0.0,
        mean_along_slope_deg=(sum(alongs) / len(alongs)) if alongs else 0.0,
        max_cross_slope_deg=max(crosses_finite) if crosses_finite else math.nan,
        n_cross_slope_undefined=n_undef,
        n_hard_violations=n_viol,
    )
    return metrics, (n_viol == 0)


def _precheck(grid: Grid, start: Cell, goal: Cell) -> str | None:
    """Return a rejection reason if start/goal are invalid, else None."""
    for label, cell in (("start", start), ("goal", goal)):
        if not grid.in_bounds(cell):
            return f"{label} cell {cell} is out of bounds {grid.shape}"
        if not grid.is_known(cell):
            return f"{label} cell {cell} is unknown/nodata (A4: unknown = blocked)"
    return None


# --------------------------------------------------------------------------- #
# Three-route comparison
# --------------------------------------------------------------------------- #
_ROUTE_NAMES: dict[Policy, str] = {
    "shortest": "Shortest (3-D distance)",
    "slope_aware": "Slope-aware",
    "full_constraint": "Full-constraint (mission)",
}


def compare_routes(
    grid: Grid, start: Cell, goal: Cell, config: PlannerConfig | None = None
) -> RouteComparison:
    """Plan all three policies to one destination and explain the differences."""
    cfg = config or PlannerConfig()
    g_col, g_row = terrain_gradient(grid)

    bad = _precheck(grid, start, goal)
    if bad is not None:
        routes = [
            RouteResult(
                name=_ROUTE_NAMES[p],
                policy=p,
                feasible=False,
                mission_feasible=False,
                reason=bad,
                path=None,
                metrics=None,
            )
            for p in POLICIES
        ]
        return RouteComparison(
            start=start, goal=goal, config=cfg, routes=routes, summary=bad
        )

    results: list[RouteResult] = []
    for policy in POLICIES:
        path, _ = plan_route(grid, start, goal, policy, cfg)
        if path is None:
            reason = (
                "no feasible path: goal unreachable under hard exclusions "
                "(slope / cross-slope / nodata)"
                if policy == "full_constraint"
                else "no feasible path: goal unreachable (blocked by nodata)"
            )
            results.append(
                RouteResult(
                    name=_ROUTE_NAMES[policy],
                    policy=policy,
                    feasible=False,
                    mission_feasible=False,
                    reason=reason,
                    path=None,
                    metrics=None,
                )
            )
            continue

        metrics, mission_ok = _evaluate_path(grid, g_col, g_row, path, cfg)
        reason = _reason_for(policy, mission_ok, metrics, cfg)
        results.append(
            RouteResult(
                name=_ROUTE_NAMES[policy],
                policy=policy,
                feasible=True,
                mission_feasible=mission_ok,
                reason=reason,
                path=path,
                metrics=metrics,
            )
        )

    return RouteComparison(
        start=start, goal=goal, config=cfg, routes=results, summary=_summarize(results)
    )


def _reason_for(
    policy: Policy, mission_ok: bool, m: RouteMetrics, cfg: PlannerConfig
) -> str:
    if policy == "full_constraint":
        return "mission route: satisfies all hard exclusions"
    if mission_ok:
        return "feasible and incidentally satisfies the mission hard exclusions"
    bits = []
    if m.max_along_slope_deg > cfg.max_slope_deg:
        bits.append(
            f"along-slope up to {m.max_along_slope_deg:.1f} deg "
            f"(> {cfg.max_slope_deg:.0f} deg limit)"
        )
    finite_cross = not math.isnan(m.max_cross_slope_deg)
    if finite_cross and m.max_cross_slope_deg > cfg.max_cross_slope_deg:
        bits.append(
            f"cross-slope up to {m.max_cross_slope_deg:.1f} deg "
            f"(> {cfg.max_cross_slope_deg:.0f} deg limit)"
        )
    if m.n_cross_slope_undefined:
        bits.append(f"{m.n_cross_slope_undefined} edge(s) adjacent to nodata")
    detail = "; ".join(bits) if bits else "hard-exclusion violations"
    return (
        f"REJECTED for mission use: {m.n_hard_violations} edge(s) violate hard "
        f"exclusions - {detail}"
    )


def _summarize(results: list[RouteResult]) -> str:
    by = {r.policy: r for r in results}
    mission = by["full_constraint"]
    shortest = by["shortest"]
    if not mission.feasible:
        return f"No mission-feasible route: {mission.reason}."
    if not shortest.feasible:
        return "Mission route found; shortest route blocked by nodata."
    sh = shortest.metrics
    mi = mission.metrics
    assert sh is not None and mi is not None
    extra = ""
    if sh.length_3d_m > 0:
        pct = 100.0 * (mi.length_3d_m - sh.length_3d_m) / sh.length_3d_m
        extra = f" Mission route is {pct:+.1f}% vs shortest by length."
    if shortest.mission_feasible:
        return (
            "Shortest route already satisfies the hard exclusions; all three "
            "policies agree here." + extra
        )
    return (
        f"Shortest route is NOT mission-feasible ({shortest.metrics.n_hard_violations} "
        f"hard-exclusion edge(s)); the mission route detours around them." + extra
    )
