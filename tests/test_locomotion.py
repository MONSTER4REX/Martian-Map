import pytest

from marswalk.physics.locomotion import (
    LocomotionParams,
    activity_factor,
    edge_energy_j,
    edge_time_s,
    speed_ms,
)


def test_speed_decreases_and_floors():
    p = LocomotionParams()
    assert speed_ms(0.0, p) == pytest.approx(p.level_speed_ms)
    assert speed_ms(5.0, p) < speed_ms(0.0, p)
    assert speed_ms(80.0, p) == pytest.approx(p.min_speed_ms)  # floored
    assert speed_ms(-5.0, p) == speed_ms(5.0, p)  # sign-independent


def test_edge_time_flat():
    p = LocomotionParams(level_speed_ms=2.0)
    # flat 10 m at 2 m/s -> 5 s
    assert edge_time_s(0.0, 10.0, p) == pytest.approx(5.0)
    # a slope takes longer than the flat equivalent horizontal distance
    assert edge_time_s(5.0, 10.0, p) > edge_time_s(0.0, 10.0, p)


def test_activity_factor_monotone():
    p = LocomotionParams()
    assert activity_factor(0.0, p) == pytest.approx(1.0)
    assert activity_factor(10.0, p) > 1.0
    assert activity_factor(20.0, p) > activity_factor(10.0, p)


def test_energy_climb_costs_more_than_descent():
    p = LocomotionParams()
    up = edge_energy_j(5.0, 10.0, p)
    down = edge_energy_j(-5.0, 10.0, p)
    level = p.cot_level * p.total_mass_kg * p.gravity_ms2 * 10.0
    assert down == pytest.approx(level)  # no credit for descent
    assert up > down
