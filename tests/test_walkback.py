import math

import numpy as np
import pytest

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.walkback import (
    WalkbackScenario,
    check_walkback,
    standard_scenarios,
    time_to_haven_field,
)
from marswalk.planner.grid import Grid
from marswalk.terrain.synthetic import punch_nodata


def _flat(rows=3, cols=60, cellsize=2.0):
    return Grid(np.zeros((rows, cols)), cellsize)


def test_time_field_zero_at_haven_and_grows():
    grid = _flat()
    field = time_to_haven_field(grid, [(1, 0)])  # default speed 1 m/s, cell 2 m
    assert field[1, 0] == 0.0
    assert field[1, 5] == pytest.approx(10.0)  # 5 edges * 2 s
    assert field[1, 59] == pytest.approx(118.0)
    # farther is never closer
    assert field[1, 10] > field[1, 5]


def test_time_field_inf_when_walled_off():
    grid = _flat()
    grid = punch_nodata(grid, [(r, 30) for r in range(grid.nrows)])  # full nodata wall
    field = time_to_haven_field(grid, [(1, 0)])
    assert math.isinf(field[1, 59])
    assert field[1, 10] < math.inf


def test_rover_loss_decision_flip():
    # Flat corridor; base at one end, rover at the far end (= the target).
    # A tight consumable budget: with the rover as a haven the walk-back is
    # always short enough; lose the rover and the far half of the route can no
    # longer get back to base in time.
    grid = _flat(rows=3, cols=60, cellsize=2.0)
    path = [(1, c) for c in range(60)]
    base, rover = (1, 0), (1, 59)
    cons = ConsumablesModel(items=[Consumable(name="o2", capacity=200.0, rate_per_s=1.0)])

    nominal, rover_loss, suit = standard_scenarios(base, rover)

    rep_nom = check_walkback(grid, path, nominal, cons, reserve_s=20.0)
    rep_rl = check_walkback(grid, path, rover_loss, cons, reserve_s=20.0)
    rep_suit = check_walkback(grid, path, suit, cons, reserve_s=20.0)

    assert rep_nom.feasible, rep_nom.reason
    assert rep_nom.min_margin_s > 0

    assert not rep_rl.feasible
    assert rep_rl.first_violation_index is not None
    assert rep_rl.first_violation_index > 0
    assert "VIOLATION" in rep_rl.reason

    # Suit degraded (slower + faster depletion) also fails.
    assert not rep_suit.feasible


def test_reserve_monotonicity():
    grid = _flat(cols=60)
    path = [(1, c) for c in range(60)]
    cons = ConsumablesModel(items=[Consumable(name="o2", capacity=200.0, rate_per_s=1.0)])
    nominal = standard_scenarios((1, 0), (1, 59))[0]
    small = check_walkback(grid, path, nominal, cons, reserve_s=20.0)
    big = check_walkback(grid, path, nominal, cons, reserve_s=20.0 + 1000.0)
    assert small.min_margin_s > big.min_margin_s
    assert small.feasible and not big.feasible


def test_unreachable_haven_reported():
    # Crew is on the far side of a nodata wall; base (near side) can't be
    # reached on foot. The outbound route stays on the reachable far side.
    grid = _flat(cols=40)
    grid = punch_nodata(grid, [(r, 20) for r in range(grid.nrows)])
    path = [(1, c) for c in range(21, 40)]  # far side only, all known cells
    cons = ConsumablesModel(items=[Consumable(name="o2", capacity=1e6, rate_per_s=1.0)])
    scen = WalkbackScenario(name="base_only", havens=[(1, 0)])
    rep = check_walkback(grid, path, scen, cons, reserve_s=0.0)
    assert not rep.feasible
    assert "no feasible walk-back path" in rep.reason


def test_path_through_nodata_rejected():
    grid = _flat(cols=40)
    grid = punch_nodata(grid, [(1, 20)])
    path = [(1, c) for c in range(40)]  # crosses the nodata cell
    cons = ConsumablesModel(items=[Consumable(name="o2", capacity=1e6, rate_per_s=1.0)])
    scen = WalkbackScenario(name="x", havens=[(1, 0)])
    with pytest.raises(ValueError, match="unknown"):
        check_walkback(grid, path, scen, cons)
