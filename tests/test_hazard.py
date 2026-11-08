"""Rock-hazard layer + hazard-aware routing + the §3 rock-ablation gate.

Offline, synthetic flat terrain so slope never forces a detour — any detour is
driven purely by the rock-hazard layer, which is what the gate must isolate.
"""

import numpy as np

from marswalk.planner.grid import Grid
from marswalk.planner.hazard import (
    Rock,
    density_to_hazard,
    rock_exposure,
    rock_hazard_field,
)
from marswalk.planner.hazard_routes import plan_hazard_route
from marswalk.planner.routes import plan_route
from marswalk.validation.ablation import rock_ablation


def _flat(n=30, cs=2.0):
    return Grid(np.zeros((n, n), dtype=float), cs)


def _wall_rocks(col=15, rows=range(8, 23), diameter_m=3.0):
    # a vertical wall of big rocks across the straight row-15 traverse
    return [Rock(row=float(r), col=float(col), diameter_m=diameter_m) for r in rows]


def test_hazard_field_bounded_and_filtered():
    grid = _flat()
    rocks = [Rock(row=15, col=15, diameter_m=4.0), Rock(row=5, col=5, diameter_m=0.5)]
    h = rock_hazard_field(grid, rocks, min_diameter_m=1.5, buffer_m=2.0)
    assert h.shape == grid.shape
    assert 0.0 <= h.min() and h.max() <= 1.0
    assert h[15, 15] == 1.0            # at a big rock's centre
    assert h[5, 5] == 0.0             # small rock filtered out
    assert h[0, 29] == 0.0            # far from any rock


def test_rock_exposure_higher_through_rocks():
    grid = _flat()
    h = rock_hazard_field(grid, _wall_rocks(), min_diameter_m=1.5)
    through = [(15, c) for c in range(30)]          # straight through the wall
    around = [(0, c) for c in range(30)]            # far from the wall
    assert rock_exposure(grid, h, through) > rock_exposure(grid, h, around)
    assert rock_exposure(grid, h, [(0, 0)]) == 0.0  # degenerate path


def test_beta0_matches_slope_only():
    grid = _flat()
    h = rock_hazard_field(grid, _wall_rocks(), min_diameter_m=1.5)
    base = plan_route(grid, (15, 0), (15, 29), "full_constraint")[0]
    beta0 = plan_hazard_route(grid, h, (15, 0), (15, 29), beta=0.0)[0]
    assert beta0 == base              # beta=0 reproduces the slope-only route


def test_rock_aware_detours_and_cuts_exposure():
    grid = _flat()
    h = rock_hazard_field(grid, _wall_rocks(), min_diameter_m=1.5, buffer_m=2.0)
    straight = plan_hazard_route(grid, h, (15, 0), (15, 29), beta=0.0)[0]
    aware = plan_hazard_route(grid, h, (15, 0), (15, 29), beta=8.0)[0]
    assert aware != straight
    assert rock_exposure(grid, h, aware) < rock_exposure(grid, h, straight)


def test_rock_ablation_gate_passes_with_boulder_field():
    grid = _flat()
    h = rock_hazard_field(grid, _wall_rocks(), min_diameter_m=1.5, buffer_m=2.0)
    rep = rock_ablation(grid, h, (15, 0), (15, 29), betas=(0.0, 2.0, 8.0))
    assert rep.gate_passed
    assert rep.exposure_reduction_frac > 0.0
    assert rep.by_beta(8.0).changed_vs_slope_only
    assert rep.by_beta(0.0).rock_exposure >= rep.by_beta(8.0).rock_exposure


def test_rock_ablation_gate_not_met_without_rocks():
    grid = _flat()
    h = np.zeros(grid.shape)                     # no rocks -> no reason to detour
    rep = rock_ablation(grid, h, (15, 0), (15, 29), betas=(0.0, 8.0))
    assert not rep.gate_passed
    assert not rep.by_beta(8.0).changed_vs_slope_only


def test_density_to_hazard_normalises():
    d = np.array([[0.0, 5.0], [10.0, np.nan]])
    h = density_to_hazard(d)
    assert h.max() == 1.0 and h.min() == 0.0
    assert np.isfinite(h).all()
    assert h[1, 0] == 1.0 and h[0, 1] == 0.5     # scaled by the field max (10)
