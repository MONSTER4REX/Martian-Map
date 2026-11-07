"""C8 geometry: line-of-sight visibility, illumination and comms coverage.

Offline on a local ENU frame (meters): a surface cell is at
(x=col*cellsize, y=row*cellsize, z=elevation). Targets (sun, relay orbiters)
are points in the same frame; `ephemeris` provides synthetic ones. Real SPICE
state vectors (spiceypy + NAIF kernels) plug in via `geometry.spice`, which is
guarded and UNTESTED ON REAL DATA.

Conventions: east = +x (+col), north = +y (+row); azimuth measured from north
toward east; elevation is the angle above the local horizontal.
"""

from marswalk.geometry.ephemeris import suggest_relay_pass, sun_direction_point
from marswalk.geometry.visibility import (
    cell_xyz,
    comms_coverage,
    elevation_azimuth,
    illumination_mask,
    is_visible,
    line_of_sight,
)

__all__ = [
    "cell_xyz",
    "elevation_azimuth",
    "line_of_sight",
    "is_visible",
    "illumination_mask",
    "comms_coverage",
    "sun_direction_point",
    "suggest_relay_pass",
]
