"""Pure service layer: build terrain, plan the three routes, probe a cell.

No web framework here, so it is fully testable offline. FastAPI (`app.py`) just
calls these functions.
"""

from __future__ import annotations

import math

import numpy as np

from marswalk.api.schemas import ProbeRequest, ProbeResponse, RouteRequest, TerrainSpec
from marswalk.data.derivatives import aspect_degrees, hillshade, slope_degrees
from marswalk.planner.grid import Grid
from marswalk.planner.routes import RouteComparison, compare_routes
from marswalk.terrain.synthetic import random_smooth_terrain, ridge_world


def build_grid(spec: TerrainSpec) -> Grid:
    if spec.kind == "ridge":
        return ridge_world(
            spec.nrows, spec.ncols, spec.cellsize,
            ridge_height_m=spec.ridge_height_m,
            ridge_half_width=spec.ridge_half_width,
            gap_col=spec.gap_col,
        )
    if spec.kind == "random":
        return random_smooth_terrain(
            spec.nrows, spec.ncols, spec.cellsize,
            seed=spec.seed, relief_m=spec.relief_m, passes=spec.passes,
        )
    if spec.kind == "inline":
        if spec.elevation is None:
            raise ValueError("inline terrain requires 'elevation'")
        arr = np.array(
            [[np.nan if v is None else float(v) for v in row] for row in spec.elevation],
            dtype=float,
        )
        return Grid(arr, spec.cellsize)
    raise ValueError(f"unknown terrain kind {spec.kind!r}")


def plan_three_routes(req: RouteRequest) -> RouteComparison:
    grid = build_grid(req.terrain)
    return compare_routes(grid, tuple(req.start), tuple(req.goal), req.config)


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def probe(req: ProbeRequest) -> ProbeResponse:
    grid = build_grid(req.terrain)
    cell = tuple(req.cell)
    in_bounds = grid.in_bounds(cell)
    known = grid.is_known(cell)
    if not in_bounds:
        return ProbeResponse(
            cell=cell, in_bounds=False, known=False,
            elevation_m=None, slope_deg=None, aspect_deg=None, hillshade=None,
        )
    elev = _finite_or_none(grid.elevation[cell])
    slope = _finite_or_none(slope_degrees(grid)[cell]) if known else None
    aspect = _finite_or_none(aspect_degrees(grid)[cell]) if known else None
    hs = _finite_or_none(hillshade(grid)[cell]) if known else None
    return ProbeResponse(
        cell=cell, in_bounds=True, known=known,
        elevation_m=elev, slope_deg=slope, aspect_deg=aspect, hillshade=hs,
    )
