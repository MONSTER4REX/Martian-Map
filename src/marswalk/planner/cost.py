"""Edge geometry and the baseline slope penalty (Assumptions A6, A7).

The slope penalty is a PLACEHOLDER, not validated locomotion energy. P4 (C5)
owns real energy/consumables. The only hard contract here is: the multiplier is
>= 1 for all slopes, which keeps the straight-line 3-D distance an admissible
A* heuristic (Assumption A10).
"""

from __future__ import annotations

import math


def horizontal_distance(dr: int, dc: int, cellsize: float) -> float:
    """Planimetric distance of an offset, in meters."""
    return cellsize * math.hypot(dr, dc)


def distance_3d(dz: float, horizontal: float) -> float:
    """True 3-D segment length from vertical drop and horizontal distance."""
    return math.hypot(dz, horizontal)


def slope_multiplier(along_slope_deg: float, alpha: float = 1.0) -> float:
    """Monotone, >= 1 penalty growing with along-path slope.

    ``multiplier = 1 + alpha * |slope_deg| / 10`` — so with alpha=1 a 10 deg
    slope doubles the effective cost. Placeholder shape; see module docstring.
    """
    return 1.0 + alpha * abs(along_slope_deg) / 10.0
