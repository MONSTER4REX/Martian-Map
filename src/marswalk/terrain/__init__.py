"""Synthetic terrain for offline tests and demos (no real data in the sandbox)."""

from marswalk.terrain.synthetic import (
    inclined_plane,
    punch_nodata,
    random_smooth_terrain,
    ridge_world,
)

__all__ = ["random_smooth_terrain", "inclined_plane", "ridge_world", "punch_nodata"]
