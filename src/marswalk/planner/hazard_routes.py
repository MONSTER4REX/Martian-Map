"""Hazard-aware routing: the full-constraint planner plus a rock-hazard cost.

Edge cost = (full-constraint slope cost) x (1 + beta * edge_hazard), where
``edge_hazard`` is the mean rock hazard of the two cells (``hazard`` in [0, 1]
from ``hazard.rock_hazard_field``). ``beta`` = 0 reproduces the slope-only
mission route exactly; larger ``beta`` trades distance for lower rock exposure.
``max_hazard`` optionally hard-excludes cells above a hazard (an impassable
boulder field).

The hazard multiplier is >= 1, so the straight-line 3-D heuristic stays
admissible and A* remains optimal for the chosen cost. This is the P2->P3 link
the §3 gate tests via ``validation.ablation.rock_ablation``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from marswalk.planner.astar import astar
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.cost import horizontal_distance
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig, _edge_cost, _make_heuristic
from marswalk.planner.slope import edge_along_slope_deg, edge_cross_slope_deg, terrain_gradient


def _hazard_neighbors(
    grid: Grid,
    hazard: np.ndarray,
    g_col: np.ndarray,
    g_row: np.ndarray,
    cfg: PlannerConfig,
    beta: float,
    max_hazard: float | None,
) -> Callable[[Cell], Iterable[tuple[Cell, float]]]:
    offsets = neighbor_offsets(cfg.connectivity)
    cs = grid.cellsize
    elev = grid.elevation

    def neighbors(node: Cell) -> Iterable[tuple[Cell, float]]:
        r, c = node
        za = elev[r, c]
        ha = float(hazard[r, c])
        for dr, dc in offsets:
            b = (r + dr, c + dc)
            if not grid.in_bounds(b):
                continue
            zb = elev[b]
            if not math.isfinite(zb):
                continue
            hb = float(hazard[b])
            if max_hazard is not None and hb > max_hazard:
                continue  # hard-exclude an impassable boulder field
            horiz = horizontal_distance(dr, dc, cs)
            dz = float(zb - za)
            along = edge_along_slope_deg(dz, horiz)
            cross = edge_cross_slope_deg(g_col, g_row, node, b)
            base = _edge_cost("full_constraint", dz, horiz, along, cross, cfg)
            if base is None:
                continue
            yield b, base * (1.0 + beta * 0.5 * (ha + hb))

    return neighbors


def plan_hazard_route(
    grid: Grid,
    hazard: np.ndarray,
    start: Cell,
    goal: Cell,
    config: PlannerConfig | None = None,
    beta: float = 1.0,
    max_hazard: float | None = None,
) -> tuple[list[Cell] | None, float]:
    """Min-cost full-constraint route under a rock-hazard penalty.

    ``beta`` weights rock avoidance vs distance; ``max_hazard`` hard-excludes
    cells above that hazard. Returns (path, total_cost); path is None if the
    goal is unreachable under the constraints.
    """
    if hazard.shape != grid.shape:
        raise ValueError(f"hazard shape {hazard.shape} != grid shape {grid.shape}")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    cfg = config or PlannerConfig()
    g_col, g_row = terrain_gradient(grid)
    neighbors = _hazard_neighbors(grid, hazard, g_col, g_row, cfg, beta, max_hazard)
    heuristic = _make_heuristic(grid, goal)
    return astar(start, goal, neighbors, heuristic)
