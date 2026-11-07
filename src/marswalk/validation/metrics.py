"""Route comparison metrics (meters, counts). All operate on cell paths."""

from __future__ import annotations

import math

from marswalk.planner.cost import distance_3d, horizontal_distance
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.slope import edge_along_slope_deg


def path_cells(path: list[Cell]) -> set[Cell]:
    return set(path)


def jaccard(a: list[Cell], b: list[Cell]) -> float:
    """Jaccard similarity of the two routes' cell sets in [0, 1]."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def path_length_3d(grid: Grid, path: list[Cell]) -> float:
    cs = grid.cellsize
    elev = grid.elevation
    total = 0.0
    for p, q in zip(path[:-1], path[1:], strict=True):
        horiz = horizontal_distance(q[0] - p[0], q[1] - p[1], cs)
        total += distance_3d(float(elev[q] - elev[p]), horiz)
    return total


def hazard_exposure(grid: Grid, path: list[Cell], max_slope_deg: float = 25.0) -> int:
    """Number of edges whose along-slope exceeds a threshold (exposure count)."""
    cs = grid.cellsize
    elev = grid.elevation
    n = 0
    for p, q in zip(path[:-1], path[1:], strict=True):
        horiz = horizontal_distance(q[0] - p[0], q[1] - p[1], cs)
        along = edge_along_slope_deg(float(elev[q] - elev[p]), horiz)
        if along > max_slope_deg:
            n += 1
    return n


def _euclid(grid: Grid, a: Cell, b: Cell) -> float:
    horiz = grid.cellsize * math.hypot(a[0] - b[0], a[1] - b[1])
    return distance_3d(float(grid.elevation[a] - grid.elevation[b]), horiz)


def discrete_frechet(grid: Grid, p: list[Cell], q: list[Cell]) -> float:
    """Discrete Fréchet distance (m) between two cell paths (iterative DP)."""
    n, m = len(p), len(q)
    if n == 0 or m == 0:
        return math.inf
    ca = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            d = _euclid(grid, p[i], q[j])
            if i == 0 and j == 0:
                ca[i][j] = d
            elif i == 0:
                ca[i][j] = max(ca[0][j - 1], d)
            elif j == 0:
                ca[i][j] = max(ca[i - 1][0], d)
            else:
                ca[i][j] = max(min(ca[i - 1][j], ca[i - 1][j - 1], ca[i][j - 1]), d)
    return ca[n - 1][m - 1]
