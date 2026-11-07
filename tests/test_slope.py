import math

import numpy as np
import pytest

from marswalk.planner.slope import (
    edge_along_slope_deg,
    edge_cross_slope_deg,
    slope_degrees,
    terrain_gradient,
)
from marswalk.terrain.synthetic import inclined_plane, punch_nodata, random_smooth_terrain


def test_flat_terrain_zero_slope():
    g = random_smooth_terrain(10, 10, relief_m=0.0, seed=1)  # relief 0 -> flat
    s = slope_degrees(g)
    assert np.allclose(s, 0.0)


def test_inclined_plane_slope_matches():
    deg = 12.0
    g = inclined_plane(20, 20, cellsize=2.0, slope_deg=deg)
    s = slope_degrees(g)
    # interior cells (central differences) should equal the plane slope
    assert np.allclose(s[1:-1, 1:-1], deg, atol=1e-6)


def test_edge_along_slope():
    # 1 m rise over 1 m run -> 45 deg
    assert edge_along_slope_deg(1.0, 1.0) == pytest.approx(45.0)
    assert edge_along_slope_deg(0.0, 3.0) == pytest.approx(0.0)
    assert edge_along_slope_deg(-5.0, 5.0) == pytest.approx(45.0)  # sign-independent


def test_cross_slope_on_plane():
    # Plane tilts along +row. Travelling across-slope (same row) sees the full
    # slope as cross-slope; travelling straight up/down-slope sees ~0 cross.
    deg = 10.0
    g = inclined_plane(20, 20, cellsize=2.0, slope_deg=deg)
    g_col, g_row = terrain_gradient(g)
    across = edge_cross_slope_deg(g_col, g_row, (10, 10), (10, 11))  # same row
    along = edge_cross_slope_deg(g_col, g_row, (10, 10), (11, 10))  # down-slope
    assert across == pytest.approx(deg, abs=1e-6)
    assert along == pytest.approx(0.0, abs=1e-6)


def test_cross_slope_nan_near_nodata():
    g = inclined_plane(10, 10, slope_deg=8.0)
    g = punch_nodata(g, [(5, 5)])
    g_col, g_row = terrain_gradient(g)
    # (5,4)->(5,6) straddles the nodata cell; its gradient is NaN.
    val = edge_cross_slope_deg(g_col, g_row, (5, 4), (4, 4))
    assert math.isnan(val)
