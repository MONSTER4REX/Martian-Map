"""Request/response models for the planner API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from marswalk.planner.grid import Cell
from marswalk.planner.routes import PlannerConfig


class TerrainSpec(BaseModel):
    """How the server obtains the DEM. Demo/synthetic only in this build."""

    kind: Literal["ridge", "random", "inline"] = "ridge"
    nrows: int = 40
    ncols: int = 40
    cellsize: float = Field(default=2.0, gt=0)
    # ridge
    ridge_height_m: float = 2.0
    ridge_half_width: int = 1
    gap_col: int | None = None
    # random
    seed: int = 0
    relief_m: float = 8.0
    passes: int = 8
    # inline (null = nodata)
    elevation: list[list[float | None]] | None = None


class RouteRequest(BaseModel):
    terrain: TerrainSpec = TerrainSpec()
    start: Cell
    goal: Cell
    config: PlannerConfig = PlannerConfig()


class ProbeRequest(BaseModel):
    terrain: TerrainSpec = TerrainSpec()
    cell: Cell


class ProbeResponse(BaseModel):
    cell: Cell
    in_bounds: bool
    known: bool
    elevation_m: float | None
    slope_deg: float | None
    aspect_deg: float | None
    hillshade: float | None
