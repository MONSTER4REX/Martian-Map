"""Rock-hazard raster: turn detected rocks into a per-cell routing hazard.

This is the bridge from the rock detector (P2) to the planner's cost (P3). A
detector gives rocks (map position + diameter); this builds a hazard field in
[0, 1] over the grid and measures a route's *rock exposure*. The planner then
routes with a hazard-aware cost (`hazard_routes.py`), and the ablation
(`validation.ablation.rock_ablation`) checks the MARSWALK_CONTEXT.md §3 gate:
rock-aware routes must reduce rock exposure vs slope-only to ship.

Model (documented, PLACEHOLDER magnitudes):
* Only rocks with ``diameter_m >= min_diameter_m`` count (the gate size is
  1.5 m). Each paints a disk of hazard 1.0 within its footprint radius, decaying
  linearly to 0 over a ``buffer_m`` clearance margin. Cells take the max over
  nearby rocks (worst rock dominates), so the field is in [0, 1].
* Unknown (NaN-elevation) cells get hazard 0 here; the planner blocks them
  separately. The hazard field carries NO slope information — it is purely the
  rock layer, kept separable so the ablation isolates its effect.

Units: ``diameter_m``/``buffer_m`` in metres; grid ``cellsize`` in m/cell.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

from marswalk.planner.cost import horizontal_distance
from marswalk.planner.grid import Cell, Grid


class Rock(BaseModel):
    """A detected rock in grid coordinates (fractional cells allowed)."""

    row: float
    col: float
    diameter_m: float = Field(gt=0)


def rock_hazard_field(
    grid: Grid,
    rocks: list[Rock],
    min_diameter_m: float = 1.5,
    buffer_m: float = 2.0,
) -> np.ndarray:
    """Per-cell rock hazard in [0, 1]. ``min_diameter_m`` filters small rocks;
    ``buffer_m`` is the soft clearance margin around each rock's footprint."""
    h = np.zeros(grid.shape, dtype=float)
    cs = grid.cellsize
    buf_cells = max(buffer_m / cs, 0.0)
    for rk in rocks:
        if rk.diameter_m < min_diameter_m:
            continue
        core = 0.5 * rk.diameter_m / cs          # footprint radius in cells
        reach = core + buf_cells                 # hazard reaches 0 here
        r0 = max(int(math.floor(rk.row - reach)), 0)
        r1 = min(int(math.ceil(rk.row + reach)), grid.nrows - 1)
        c0 = max(int(math.floor(rk.col - reach)), 0)
        c1 = min(int(math.ceil(rk.col + reach)), grid.ncols - 1)
        if r1 < r0 or c1 < c0:
            continue
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        dist = np.hypot(rr - rk.row, cc - rk.col)
        if buf_cells > 0:
            contrib = np.clip(1.0 - np.clip(dist - core, 0.0, None) / buf_cells, 0.0, 1.0)
        else:
            contrib = (dist <= core).astype(float)
        np.maximum(h[r0:r1 + 1, c0:c1 + 1], contrib, out=h[r0:r1 + 1, c0:c1 + 1])
    return h


def rock_exposure(grid: Grid, hazard: np.ndarray, path: list[Cell]) -> float:
    """Route's integrated rock hazard (hazard·m): sum over edges of the mean
    endpoint hazard times the edge's horizontal length. 0 for a <2-cell path."""
    if len(path) < 2:
        return 0.0
    cs = grid.cellsize
    total = 0.0
    for p, q in zip(path[:-1], path[1:], strict=True):
        horiz = horizontal_distance(q[0] - p[0], q[1] - p[1], cs)
        total += 0.5 * (float(hazard[p]) + float(hazard[q])) * horiz
    return total


def density_to_hazard(density: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Normalise a boulder-count/density raster to a [0, 1] hazard field.

    ``scale`` is the density mapped to hazard 1.0; if None, the field's own max
    is used. NaNs become 0. Use this when the detector gives a density raster
    rather than discrete rocks.
    """
    d = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    s = scale if scale is not None else (d.max() if d.max() > 0 else 1.0)
    return np.clip(d / s, 0.0, 1.0)
