"""Survival envelope / no-go zones (the hero capability).

A tight consumables budget makes the budget the binding constraint, so only a
*partial* region is survivable and the no-go zones are meaningful. Flat terrain
isolates the walk-back/reach effect from slope.
"""

import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.survival import (
    compare_survival,
    reach_time_field,
    survival_envelope,
)
from marswalk.physics.walkback import WalkbackScenario, standard_scenarios
from marswalk.planner.grid import Grid

CS = 2.0
# budget 91 s, reserve 20 -> ~71 s for reach+haven: partial envelope on this grid
TIGHT = ConsumablesModel(items=[Consumable(name="o2", capacity=91.0, rate_per_s=1.0)])


def _flat(n=40):
    return Grid(np.zeros((n, n), dtype=float), CS)


def test_reach_time_field_grows_from_start():
    grid = _flat()
    reach = reach_time_field(grid, (20, 20))
    assert reach[20, 20] == 0.0
    assert reach[20, 25] < reach[20, 30]          # farther = longer
    assert np.isfinite(reach).all()               # flat grid fully reachable


def test_reach_blocked_by_nodata_wall():
    a = np.zeros((20, 20)); a[:, 10] = np.nan     # a wall of unknown cells
    grid = Grid(a, CS)
    reach = reach_time_field(grid, (10, 0))
    assert reach[10, 0] == 0.0
    assert not np.isfinite(reach[10, 19])         # far side unreachable


def test_envelope_is_partial_with_tight_budget():
    grid = _flat()
    nominal = standard_scenarios((0, 0), (39, 39))[0]
    env = survival_envelope(grid, (20, 20), nominal, TIGHT, reserve_s=20.0)
    assert 0.0 < env.survivable_fraction < 1.0    # some safe, some no-go
    assert env.survivable[20, 20]                 # start is survivable
    assert env.no_go().sum() > 0
    # a cell survivable => reach + haven + reserve <= budget
    r, c = np.argwhere(env.survivable)[0]
    assert env.reach_s[r, c] + env.haven_s[r, c] + env.reserve_s <= env.budget_s + 1e-6


def test_rover_loss_and_suit_degraded_shrink_the_world():
    grid = _flat()
    scen = standard_scenarios((0, 0), (39, 39))  # nominal, rover_loss, suit_degraded
    cmp = compare_survival(grid, (20, 20), scen, TIGHT, reserve_s=20.0)
    nom, rover_loss, degraded = (e.n_survivable for e in cmp.envelopes)
    assert rover_loss <= nom                       # losing a haven can only shrink it
    assert degraded < nom                          # slower + faster burn shrinks it
    shrink = cmp.shrink_vs_first()
    assert shrink[0]["frac_lost_vs_nominal"] == 0.0
    assert shrink[2]["frac_lost_vs_nominal"] > 0.0


def test_tighter_reserve_shrinks_envelope():
    grid = _flat()
    nominal = standard_scenarios((0, 0), (39, 39))[0]
    loose = survival_envelope(grid, (20, 20), nominal, TIGHT, reserve_s=20.0)
    tight = survival_envelope(grid, (20, 20), nominal, TIGHT, reserve_s=30.0)
    assert 0 < tight.n_survivable < loose.n_survivable


def test_unreachable_haven_yields_all_no_go():
    # haven on an isolated island (walled off) -> nobody can walk back -> all no-go
    a = np.zeros((20, 20)); a[:, 10] = np.nan
    grid = Grid(a, CS)
    scen = WalkbackScenario(name="islanded", havens=[(10, 19)])  # far side of the wall
    env = survival_envelope(grid, (10, 0), scen, TIGHT, reserve_s=20.0)
    assert env.n_survivable == 0
    assert env.no_go().sum() == env.n_known
