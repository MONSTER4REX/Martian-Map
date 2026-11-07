"""DEM derivatives: slope, aspect, hillshade, curvature. All NaN-aware.

Units: elevation m, cellsize m/cell, slope/aspect degrees, curvature 1/m.
Conventions:
* gradient from `numpy.gradient` scaled by cellsize (central differences).
* aspect: compass degrees [0, 360), 0 = +row direction ("north"), increasing
  toward +col ("east"); NaN where the surface is flat (slope 0).
* hillshade: Horn-style, 0..255, for a given sun azimuth/altitude (degrees).
Cells adjacent to nodata inherit NaN (missing stays missing).
"""

from __future__ import annotations

import numpy as np

from marswalk.planner.grid import Grid


def _gradient(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    g_row, g_col = np.gradient(grid.elevation, grid.cellsize)
    return g_col, g_row  # (d/dcol = x/east, d/drow = y/north)


def slope_degrees(grid: Grid) -> np.ndarray:
    g_col, g_row = _gradient(grid)
    return np.degrees(np.arctan(np.hypot(g_col, g_row)))


def aspect_degrees(grid: Grid) -> np.ndarray:
    """Downslope-facing compass aspect in [0, 360); NaN where flat."""
    g_col, g_row = _gradient(grid)
    mag = np.hypot(g_col, g_row)
    # Downslope direction is -gradient. Measure from +row toward +col.
    aspect = np.degrees(np.arctan2(-g_col, -g_row))
    aspect = np.mod(aspect, 360.0)
    aspect = np.where(mag > 0, aspect, np.nan)
    return aspect


def hillshade(
    grid: Grid, azimuth_deg: float = 315.0, altitude_deg: float = 45.0
) -> np.ndarray:
    """Horn hillshade in [0, 255]; NaN where elevation is unknown."""
    g_col, g_row = _gradient(grid)
    slope = np.arctan(np.hypot(g_col, g_row))
    aspect = np.arctan2(-g_col, -g_row)  # radians, same convention as above
    zenith = np.radians(90.0 - altitude_deg)
    az = np.radians(azimuth_deg)
    shaded = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0.0, 1.0) * 255.0


def curvature(grid: Grid) -> np.ndarray:
    """Laplacian curvature d2z/dx2 + d2z/dy2 (1/m); NaN-aware."""
    z = grid.elevation
    cs2 = grid.cellsize**2
    d2_row = np.full_like(z, np.nan)
    d2_col = np.full_like(z, np.nan)
    d2_row[1:-1, :] = (z[2:, :] - 2 * z[1:-1, :] + z[:-2, :]) / cs2
    d2_col[:, 1:-1] = (z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]) / cs2
    return d2_row + d2_col
