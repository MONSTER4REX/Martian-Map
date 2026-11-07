import numpy as np
import pytest

from marswalk.geometry.ephemeris import suggest_relay_pass
from marswalk.geometry.visibility import (
    comms_coverage,
    elevation_azimuth,
    illumination_mask,
    is_visible,
    line_of_sight,
)
from marswalk.planner.grid import Grid


def _flat(rows=20, cols=20, cs=2.0):
    return Grid(np.zeros((rows, cols)), cs)


def test_elevation_azimuth():
    o = (0.0, 0.0, 0.0)
    el, _ = elevation_azimuth(o, (0.0, 0.0, 100.0))
    assert el == pytest.approx(90.0)
    _, az_n = elevation_azimuth(o, (0.0, 10.0, 0.0))
    _, az_e = elevation_azimuth(o, (10.0, 0.0, 0.0))
    _, az_ne = elevation_azimuth(o, (10.0, 10.0, 0.0))
    assert az_n == pytest.approx(0.0)
    assert az_e == pytest.approx(90.0)
    assert az_ne == pytest.approx(45.0)


def test_los_clear_over_flat():
    g = _flat()
    # a relay high above the far side of the AOI
    assert line_of_sight(g, (10, 2), (38.0, 20.0, 50000.0))


def test_los_blocked_by_wall():
    z = np.zeros((20, 20))
    z[:, 10] = 200.0  # tall N-S wall
    g = Grid(z, 2.0)
    # observer west of the wall, low target east of the wall -> blocked
    assert not line_of_sight(g, (10, 2), (36.0, 20.0, 5.0))
    # a target high enough / on the same side stays visible
    assert line_of_sight(g, (10, 12), (36.0, 20.0, 5.0))


def test_is_visible_elevation_mask():
    g = _flat()
    low = (1000.0, 20.0, 30.0)  # far away, low in the sky
    el, _ = elevation_azimuth((20.0, 20.0, 0.0), low)
    assert el < 5.0
    assert not is_visible(g, (10, 10), low, min_elevation_deg=5.0)
    assert is_visible(g, (10, 10), (20.0, 20.0, 50000.0), min_elevation_deg=5.0)


def test_illumination_all_lit_and_none():
    g = _flat()
    assert illumination_mask(g, sun_azimuth_deg=180, sun_elevation_deg=30).all()
    assert not illumination_mask(g, sun_azimuth_deg=180, sun_elevation_deg=-5).any()


def test_illumination_wall_casts_shadow():
    z = np.zeros((10, 20))
    z[:, 10] = 200.0  # tall wall
    g = Grid(z, 2.0)
    lit = illumination_mask(g, sun_azimuth_deg=90, sun_elevation_deg=20)  # sun in the east
    assert not lit[5, 5]    # west of the wall: shadowed
    assert not lit[5, 9]    # just west of the wall: shadowed
    assert lit[5, 15]       # east of the wall: lit


def test_comms_coverage_range():
    g = _flat()
    track = suggest_relay_pass(g, n_steps=12, max_elevation_deg=80.0)
    cov = comms_coverage(g, track, min_elevation_deg=10.0)
    finite = cov[np.isfinite(cov)]
    assert finite.size == g.nrows * g.ncols
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    assert cov[10, 10] > 0.0  # centre sees part of the pass
