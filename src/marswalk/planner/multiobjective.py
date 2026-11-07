"""Multi-objective route search on a corridor (spec step 5, NAMOA*-style).

Returns the Pareto-optimal set of routes trading off several objectives, all
minimized, over the full-constraint traversable graph restricted to an optional
corridor mask (MARSWALK_CONTEXT.md §3: multi-objective search on a corridor
mask only). This is the gated step 5, now unlocked by the walk-back gate.

Objectives (default): distance (3-D m), time (s), hazard (slope-penalized
excess length). Any subset is selectable; all are non-negative and additive, so
the exact Pareto front is found by multi-objective label-setting (Martins'
algorithm = NAMOA* with a zero heuristic). An admissible *vector* heuristic can
be added later to prune without changing the result.

Soundness check (tests): the distance-minimizing Pareto label equals the
single-objective shortest-distance cost, and the time-minimizing label equals a
single-objective shortest-time cost (both cross-checked against NetworkX).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

from marswalk.physics.locomotion import LocomotionParams, edge_time_s
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.cost import distance_3d, horizontal_distance, slope_multiplier
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig, _edge_cost
from marswalk.planner.slope import edge_along_slope_deg, edge_cross_slope_deg, terrain_gradient

OBJECTIVES = ("distance", "time", "hazard")

Vector = tuple[float, ...]


class ParetoRoute(BaseModel):
    cost: dict[str, float]
    path: list[Cell]


class ParetoFront(BaseModel):
    objectives: list[str]
    start: Cell
    goal: Cell
    routes: list[ParetoRoute]

    def best_for(self, objective: str) -> ParetoRoute:
        """The route minimizing a single objective (ties -> first)."""
        if objective not in self.objectives:
            raise ValueError(f"{objective!r} not in {self.objectives}")
        return min(self.routes, key=lambda r: r.cost[objective])


def _dominates(a: Vector, b: Vector) -> bool:
    """True if a dominates b: a <= b componentwise and a != b."""
    le = all(x <= y for x, y in zip(a, b, strict=True))
    lt = any(x < y for x, y in zip(a, b, strict=True))
    return le and lt


def _nondominated_insert(labels: list[Vector], cand: Vector) -> bool:
    """Try to add cand to a Pareto set. Returns True if added (and prunes)."""
    for v in labels:
        if _dominates(v, cand) or v == cand:
            return False
    labels[:] = [v for v in labels if not _dominates(cand, v)]
    labels.append(cand)
    return True


def _edge_objective_builder(
    objectives: list[str], loco: LocomotionParams
) -> Callable[[float, float, float], Vector]:
    def vec(dz: float, horiz: float, along: float) -> Vector:
        dist = distance_3d(dz, horiz)
        out = []
        for o in objectives:
            if o == "distance":
                out.append(dist)
            elif o == "time":
                out.append(edge_time_s(dz, horiz, loco))
            elif o == "hazard":
                out.append(dist * (slope_multiplier(along) - 1.0))
            else:
                raise ValueError(f"unknown objective {o!r}; choose from {OBJECTIVES}")
        return tuple(out)

    return vec


def plan_pareto(
    grid: Grid,
    start: Cell,
    goal: Cell,
    objectives: tuple[str, ...] = OBJECTIVES,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    corridor: np.ndarray | None = None,
    max_expansions: int = 2_000_000,
) -> ParetoFront:
    """Exact Pareto front of routes start->goal over the corridor."""
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()
    objs = list(objectives)
    if not objs:
        raise ValueError("need at least one objective")
    edge_vec = _edge_objective_builder(objs, loco)
    g_col, g_row = terrain_gradient(grid)
    offsets = neighbor_offsets(pcfg.connectivity)
    cs = grid.cellsize
    elev = grid.elevation

    def allowed(cell: Cell) -> bool:
        if not grid.is_known(cell):
            return False
        return bool(corridor[cell]) if corridor is not None else True

    if not allowed(start) or not allowed(goal):
        return ParetoFront(objectives=objs, start=start, goal=goal, routes=[])

    # Each label: (cost_vector, node, parent_label_id). Permanent labels per node.
    perm: dict[Cell, list[Vector]] = {}
    # store parent pointers keyed by (node, cost_vector) for reconstruction
    parent: dict[tuple[Cell, Vector], tuple[Cell, Vector] | None] = {}
    zero: Vector = tuple(0.0 for _ in objs)
    parent[(start, zero)] = None

    counter = 0
    heap: list[tuple[Vector, int, Cell, Vector]] = [(zero, counter, start, zero)]
    goal_front: list[Vector] = []
    expansions = 0

    while heap:
        key, _, node, cost = heapq.heappop(heap)
        node_perm = perm.setdefault(node, [])
        if not _nondominated_insert(node_perm, cost):
            continue  # dominated label, skip
        if node == goal:
            _nondominated_insert(goal_front, cost)
            continue
        # prune: if already dominated by a goal label, no point extending
        if any(_dominates(gv, cost) for gv in goal_front):
            continue
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("Pareto search exceeded max_expansions; shrink the corridor")
        r, c = node
        za = elev[node]
        for dr, dc in offsets:
            b = (r + dr, c + dc)
            if not grid.in_bounds(b) or not allowed(b):
                continue
            zb = elev[b]
            if not math.isfinite(zb):
                continue
            horiz = horizontal_distance(dr, dc, cs)
            dz = float(zb - za)
            along = edge_along_slope_deg(dz, horiz)
            cross = edge_cross_slope_deg(g_col, g_row, node, b)
            if _edge_cost("full_constraint", dz, horiz, along, cross, pcfg) is None:
                continue
            ev = edge_vec(dz, horiz, along)
            new = tuple(x + y for x, y in zip(cost, ev, strict=True))
            # cheap pre-check against b's permanent set
            bperm = perm.get(b, [])
            if any(_dominates(v, new) or v == new for v in bperm):
                continue
            if any(_dominates(gv, new) for gv in goal_front):
                continue
            parent[(b, new)] = (node, cost)
            counter += 1
            heapq.heappush(heap, (new, counter, b, new))

    routes = [
        ParetoRoute(cost=dict(zip(objs, v, strict=True)), path=_reconstruct(parent, goal, v))
        for v in sorted(goal_front)
    ]
    return ParetoFront(objectives=objs, start=start, goal=goal, routes=routes)


def _reconstruct(
    parent: dict[tuple[Cell, Vector], tuple[Cell, Vector] | None],
    goal: Cell,
    cost: Vector,
) -> list[Cell]:
    path = [goal]
    cur: tuple[Cell, Vector] | None = (goal, cost)
    while cur is not None:
        prev = parent.get(cur)
        if prev is None:
            break
        path.append(prev[0])
        cur = prev
    path.reverse()
    return path


def corridor_mask(grid: Grid, start: Cell, goal: Cell, half_width_cells: int) -> np.ndarray:
    """A simple rectangular+buffer corridor around the start-goal segment.

    Cell is in the corridor if its perpendicular distance (in cells) to the
    start->goal line segment is <= half_width_cells. Keeps multi-objective
    search tractable (spec: corridor mask only).
    """
    rr, cc = np.mgrid[0 : grid.nrows, 0 : grid.ncols]
    s = np.array(start, dtype=float)
    g = np.array(goal, dtype=float)
    seg = g - s
    seg_len2 = float(seg @ seg)
    if seg_len2 == 0:
        dist = np.hypot(rr - s[0], cc - s[1])
    else:
        t = ((rr - s[0]) * seg[0] + (cc - s[1]) * seg[1]) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        projr = s[0] + t * seg[0]
        projc = s[1] + t * seg[1]
        dist = np.hypot(rr - projr, cc - projc)
    return dist <= half_width_cells
