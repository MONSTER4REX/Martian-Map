"""Terrain gradient, along-path slope and cross-slope (Assumptions A8, A9).

All angles in degrees. Terrain gradient is dimensionless rise/run, computed with
central differences scaled by cellsize. Cells adjacent to nodata get NaN
gradient, which propagates to NaN cross-slope and thus a hard exclusion.
"""

from __future__ import annotations

import math

import numpy as np

from marswalk.planner.grid import Grid


def terrain_gradient(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return (g_col, g_row): rise/run along +col (x) and +row (y).

    Uses numpy.gradient (central differences interior, one-sided at edges)
    scaled by cellsize. NaN elevation propagates to NaN gradient.
    """
    g_row, g_col = np.gradient(grid.elevation, grid.cellsize)
    return g_col, g_row


def slope_degrees(grid: Grid) -> np.ndarray:
    """Per-cell terrain slope magnitude in degrees: atan(|grad|)."""
    g_col, g_row = terrain_gradient(grid)
    mag = np.hypot(g_col, g_row)
    return np.degrees(np.arctan(mag))


def edge_along_slope_deg(dz: float, horizontal: float) -> float:
    """Slope of travel along an edge: atan2(|dz|, horizontal distance)."""
    return math.degrees(math.atan2(abs(dz), horizontal))


def edge_cross_slope_deg(
    g_col: np.ndarray,
    g_row: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> float:
    """Cross-slope angle for the edge a->b, from the mean endpoint gradient.

    The cross-slope is the terrain-gradient component perpendicular to the
    direction of travel. Returns NaN if either endpoint's gradient is NaN
    (i.e. adjacent to nodata) so the caller can treat it as a hard exclusion.
    """
    gx = 0.5 * (g_col[a] + g_col[b])
    gy = 0.5 * (g_row[a] + g_row[b])
    if not (math.isfinite(gx) and math.isfinite(gy)):
        return math.nan
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    norm = math.hypot(dr, dc)
    if norm == 0:
        return 0.0
    # Unit travel direction in (col, row) space = (dc, dr) / norm.
    tx, ty = dc / norm, dr / norm
    # Perpendicular component magnitude = |g x t| (2-D cross product).
    cross = abs(gx * ty - gy * tx)
    return math.degrees(math.atan(cross))
