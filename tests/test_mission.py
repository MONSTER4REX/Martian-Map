import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.walkback import check_walkback, standard_scenarios
from marswalk.planner.grid import Grid
from marswalk.planner.mission import compare_routes_with_walkback, plan_walkback_route


def _flat(rows=3, cols=40, cellsize=2.0):
    return Grid(np.zeros((rows, cols)), cellsize)


def _cons(cap):
    return ConsumablesModel(items=[Consumable(name="o2", capacity=cap, rate_per_s=1.0)])


def test_walkback_route_feasible_is_sound():
    # Nominal: rover is a haven at the far end; a walk-back-safe route exists
    # to the far goal, and it independently re-verifies as feasible.
    grid = _flat(cols=40)
    cons = _cons(120.0)
    nominal = standard_scenarios((1, 0), (1, 39))[0]
    res = plan_walkback_route(grid, (1, 0), (1, 39), nominal, cons, reserve_s=20.0)
    assert res.feasible, res.reason
    assert res.path is not None and res.path[0] == (1, 0) and res.path[-1] == (1, 39)
    assert res.min_margin_s >= 0
    # Re-verify with the independent checker.
    chk = check_walkback(grid, res.path, nominal, cons, reserve_s=20.0)
    assert chk.feasible


def test_rover_loss_makes_far_goal_infeasible():
    # Lose the rover: the far goal can no longer keep a walk-back to base.
    grid = _flat(cols=40)
    cons = _cons(120.0)
    rover_loss = standard_scenarios((1, 0), (1, 39))[1]
    res = plan_walkback_route(grid, (1, 0), (1, 39), rover_loss, cons, reserve_s=20.0)
    assert not res.feasible
    assert "unreachable within budget" in res.reason


def test_rover_loss_nearer_goal_feasible():
    # A nearer goal stays walk-back-safe even without the rover.
    grid = _flat(cols=40)
    cons = _cons(120.0)
    rover_loss = standard_scenarios((1, 0), (1, 39))[1]
    res = plan_walkback_route(grid, (1, 0), (1, 20), rover_loss, cons, reserve_s=20.0)
    assert res.feasible, res.reason
    assert res.min_margin_s >= 0


def test_scenario_decision_flip():
    # The headline (spec demo a): with the rover as a haven the far trip is
    # walk-back-safe; lose the rover and NO safe route to that goal exists.
    grid = _flat(cols=40)
    cons = _cons(120.0)
    base, rover = (1, 0), (1, 39)
    nominal, rover_loss, _ = standard_scenarios(base, rover)

    nom = compare_routes_with_walkback(grid, base, (1, 39), nominal, cons, reserve_s=20.0)
    rl = compare_routes_with_walkback(grid, base, (1, 39), rover_loss, cons, reserve_s=20.0)

    # Nominal: shortest route is already walk-back-safe and a route exists.
    assert nom.shortest_feasible_walkback
    assert nom.walkback_route.feasible

    # Rover-loss: shortest route violates walk-back AND no safe route reaches
    # the far goal at all -> the mission decision flips from "go" to "no-go".
    assert not rl.shortest_feasible_walkback
    assert not rl.walkback_route.feasible
    assert "neither" in rl.summary


def test_infeasible_start_reported():
    grid = _flat(cols=40)
    cons = _cons(10.0)  # tiny budget: can't even cover reserve + walk-back
    rover_loss = standard_scenarios((1, 0), (1, 39))[1]
    res = plan_walkback_route(grid, (1, 30), (1, 39), rover_loss, cons, reserve_s=100.0)
    assert not res.feasible
    assert "start" in res.reason


def test_unknown_goal_rejected():
    from marswalk.terrain.synthetic import punch_nodata

    grid = punch_nodata(_flat(cols=40), [(1, 39)])
    cons = _cons(120.0)
    nominal = standard_scenarios((1, 0), (1, 39))[0]
    res = plan_walkback_route(grid, (1, 0), (1, 39), nominal, cons, reserve_s=20.0)
    assert not res.feasible
    assert "unknown/nodata" in res.reason
