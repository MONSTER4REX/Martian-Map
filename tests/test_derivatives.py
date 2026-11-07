import math

import numpy as np

from marswalk.data.derivatives import aspect_degrees, curvature, hillshade, slope_degrees
from marswalk.planner.grid import Grid
from marswalk.terrain.synthetic import inclined_plane, punch_nodata, random_smooth_terrain


def test_flat_derivatives():
    g = random_smooth_terrain(10, 10, relief_m=0.0, seed=0)  # flat
    assert np.allclose(slope_degrees(g), 0.0)
    assert np.all(np.isnan(aspect_degrees(g)))  # flat -> aspect undefined
    hs = hillshade(g, azimuth_deg=315, altitude_deg=45)
    assert np.allclose(hs, math.cos(math.radians(45)) * 255.0)


def test_inclined_plane_slope_and_aspect():
    g = inclined_plane(20, 20, cellsize=2.0, slope_deg=15.0)
    assert np.allclose(slope_degrees(g)[1:-1, 1:-1], 15.0, atol=1e-6)
    asp = aspect_degrees(g)[1:-1, 1:-1]
    assert np.all((asp >= 0) & (asp < 360))
    assert np.allclose(asp, asp[0, 0])  # uniform aspect on a plane


def test_curvature_plane_is_zero():
    g = inclined_plane(20, 20, slope_deg=10.0)
    assert np.allclose(np.nan_to_num(curvature(g)[1:-1, 1:-1]), 0.0, atol=1e-9)


def test_nan_propagation():
    g = inclined_plane(10, 10, slope_deg=8.0)
    g = punch_nodata(g, [(5, 5)])
    assert math.isnan(slope_degrees(g)[5, 4])  # neighbour of nodata
    assert math.isnan(curvature(g)[5, 5])


def test_curvature_bump_is_negative_at_peak():
    z = np.zeros((11, 11))
    z[5, 5] = 4.0
    z[4:7, 4:7] += np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=float)
    g = Grid(z, 1.0)
    assert curvature(g)[5, 5] < 0  # convex up -> negative laplacian
