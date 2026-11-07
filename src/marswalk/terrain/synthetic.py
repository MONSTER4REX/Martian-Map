"""Deterministic synthetic terrain generators.

These exist only so the planner can be exercised and tested offline. They are
NOT a model of Mars. Real DEMs arrive via P1 (COG window reader); planner
behavior on real data is UNTESTED until an owner runs it.
"""

from __future__ import annotations

import numpy as np

from marswalk.planner.grid import Grid


def _smooth(a: np.ndarray, passes: int) -> np.ndarray:
    """A few passes of a 5-point mean filter (edge-padded)."""
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (
            p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[1:-1, 1:-1]
        ) / 5.0
    return a


def random_smooth_terrain(
    nrows: int,
    ncols: int,
    cellsize: float = 2.0,
    *,
    seed: int = 0,
    relief_m: float = 20.0,
    passes: int = 8,
) -> Grid:
    """Smoothed random terrain scaled to roughly ``relief_m`` peak-to-peak."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nrows, ncols))
    a = _smooth(a, passes)
    span = float(a.max() - a.min())
    if span > 0:
        a = (a - a.min()) / span * relief_m
    return Grid(a, cellsize)


def inclined_plane(
    nrows: int, ncols: int, cellsize: float = 2.0, *, slope_deg: float = 10.0
) -> Grid:
    """A uniform plane tilted along +row by ``slope_deg``."""
    rise_per_m = np.tan(np.radians(slope_deg))
    rows = np.arange(nrows)[:, None] * cellsize * rise_per_m
    a = np.broadcast_to(rows, (nrows, ncols)).astype(float).copy()
    return Grid(a, cellsize)


def ridge_world(
    nrows: int = 40,
    ncols: int = 40,
    cellsize: float = 2.0,
    *,
    ridge_height_m: float = 40.0,
    ridge_half_width: int = 2,
    gap_col: int | None = None,
) -> Grid:
    """A flat plain split by a steep E-W ridge, with one low gap/pass.

    Use for route-divergence demos: a straight line crosses the steep ridge
    (hard-slope violation), while the mission route detours through the gap.
    """
    a = np.zeros((nrows, ncols), dtype=float)
    mid = nrows // 2
    triangle = np.clip(ridge_half_width - np.abs(np.arange(nrows) - mid), 0, None)
    profile = (triangle / max(ridge_half_width, 1)) * ridge_height_m
    a += profile[:, None]
    if gap_col is None:
        gap_col = ncols // 2
    # Carve a narrow low-slope gap through the ridge.
    lo = max(gap_col - 1, 0)
    hi = min(gap_col + 2, ncols)
    a[:, lo:hi] = 0.0
    return Grid(a, cellsize)


def punch_nodata(grid: Grid, cells: list[tuple[int, int]]) -> Grid:
    """Return a copy with the given cells set to NaN (unknown)."""
    a = grid.elevation.copy()
    for r, c in cells:
        a[r, c] = np.nan
    return Grid(a, grid.cellsize)
