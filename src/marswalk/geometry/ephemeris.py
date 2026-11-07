"""Synthetic ephemeris: stand-in relay passes and sun directions.

These are NOT real geometry — they let the visibility/coverage code be tested
offline. Real orbiter/sun states come from SPICE (see geometry.spice), which an
owner wires with NAIF kernels; that path is UNTESTED ON REAL DATA.
"""

from __future__ import annotations

import math

from marswalk.geometry.visibility import XYZ
from marswalk.planner.grid import Grid


def sun_direction_point(
    grid: Grid, azimuth_deg: float, elevation_deg: float, far_m: float = 1e7
) -> XYZ:
    """A far point in the sun direction, anchored at the AOI centre."""
    cx = 0.5 * grid.ncols * grid.cellsize
    cy = 0.5 * grid.nrows * grid.cellsize
    cz = 0.0
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        cx + far_m * math.sin(az) * math.cos(el),
        cy + far_m * math.cos(az) * math.cos(el),
        cz + far_m * math.sin(el),
    )


def suggest_relay_pass(
    grid: Grid,
    n_steps: int = 12,
    altitude_m: float = 400_000.0,
    pass_azimuth_deg: float = 90.0,
    max_elevation_deg: float = 80.0,
) -> list[XYZ]:
    """A simple relay arc rising and setting across the sky over the AOI centre.

    Positions sweep elevation 0 -> max -> 0 along a fixed azimuth, at a high
    altitude, as a stand-in for a single orbiter overpass.
    """
    cx = 0.5 * grid.ncols * grid.cellsize
    cy = 0.5 * grid.nrows * grid.cellsize
    az = math.radians(pass_azimuth_deg)
    track: list[XYZ] = []
    for i in range(n_steps):
        frac = i / max(n_steps - 1, 1)
        el = math.radians(max_elevation_deg * math.sin(math.pi * frac))  # 0..max..0
        ground = altitude_m / math.tan(el) if el > 1e-6 else altitude_m * 1e3
        track.append(
            (
                cx + ground * math.sin(az),
                cy + ground * math.cos(az),
                altitude_m,
            )
        )
    return track
