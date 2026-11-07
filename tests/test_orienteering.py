import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.walkback import WalkbackScenario
from marswalk.planner.grid import Grid
from marswalk.planner.orienteering import ScienceTarget, plan_orienteering
from marswalk.terrain.synthetic import punch_nodata


def _flat(rows=11, cols=21, cellsize=2.0):
    return Grid(np.zeros((rows, cols)), cellsize)


def _cons(cap):
    return ConsumablesModel(items=[Consumable(name="o2", capacity=cap, rate_per_s=1.0)])


def _scen(base):
    return WalkbackScenario(name="base", havens=[base])


def test_visits_all_when_budget_generous():
    grid = _flat()
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 0), value=3),
        ScienceTarget(cell=(5, 20), value=5),
        ScienceTarget(cell=(0, 10), value=2),
    ]
    res = plan_orienteering(grid, base, targets, _cons(10000.0), _scen(base), reserve_s=10.0)
    assert set(res.order) == {0, 1, 2}
    assert res.total_value == 10
    assert res.total_time_s + res.reserve_s <= res.budget_s
    assert res.excluded == []


def test_budget_forces_higher_value_subset():
    grid = _flat()
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 0), value=3),   # left, round trip ~80 s
        ScienceTarget(cell=(5, 20), value=5),  # right, round trip ~80 s
    ]
    # Budget fits one side (~44 s) but not both (~84 s): pick the higher value.
    res = plan_orienteering(grid, base, targets, _cons(60.0), _scen(base), reserve_s=4.0)
    assert res.order == [1]
    assert res.total_value == 5
    assert res.total_time_s + res.reserve_s <= res.budget_s
    assert res.excluded == []  # each is individually visitable


def test_time_window_excludes_late_target():
    grid = _flat()
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 20), value=5, close_s=5.0),  # arrival ~20 s > 5
        ScienceTarget(cell=(5, 12), value=1),
    ]
    res = plan_orienteering(grid, base, targets, _cons(10000.0), _scen(base), reserve_s=10.0)
    assert 0 in res.excluded
    assert 0 not in res.order
    assert 1 in res.order


def test_walkback_excludes_far_target():
    grid = _flat()
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 20), value=5),  # arrive20 + tth20 + reserve4 = 44 > 30
        ScienceTarget(cell=(5, 12), value=1),  # arrive4 + tth4 + reserve4 = 12 <= 30
    ]
    res = plan_orienteering(grid, base, targets, _cons(30.0), _scen(base), reserve_s=4.0)
    assert 0 in res.excluded
    assert res.order == [1]


def test_unreachable_target_excluded():
    grid = _flat()
    grid = punch_nodata(grid, [(r, 15) for r in range(grid.nrows)])  # wall at col15
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 20), value=5),  # behind the wall
        ScienceTarget(cell=(5, 5), value=2),   # reachable
    ]
    res = plan_orienteering(grid, base, targets, _cons(10000.0), _scen(base), reserve_s=10.0)
    assert 0 in res.excluded
    assert res.order == [1]


def test_determinism():
    grid = _flat()
    base = (5, 10)
    targets = [
        ScienceTarget(cell=(5, 2), value=1),
        ScienceTarget(cell=(8, 14), value=2),
        ScienceTarget(cell=(2, 18), value=3),
    ]
    a = plan_orienteering(grid, base, targets, _cons(400.0), _scen(base), reserve_s=10.0)
    b = plan_orienteering(grid, base, targets, _cons(400.0), _scen(base), reserve_s=10.0)
    assert a.order == b.order
    assert a.total_value == b.total_value
