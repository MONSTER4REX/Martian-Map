import numpy as np

from marswalk.data.error_model import DEMErrorModel
from marswalk.planner.grid import Grid
from marswalk.planner.routes import PlannerConfig
from marswalk.terrain.synthetic import inclined_plane, random_smooth_terrain, ridge_world
from marswalk.validation.ablation import ablation
from marswalk.validation.metrics import (
    discrete_frechet,
    hazard_exposure,
    jaccard,
    path_length_3d,
)
from marswalk.validation.stability import route_stability


def test_jaccard():
    a = [(0, 0), (0, 1), (0, 2)]
    assert jaccard(a, a) == 1.0
    assert jaccard(a, [(5, 5)]) == 0.0
    assert 0.0 < jaccard(a, [(0, 0), (0, 1), (9, 9)]) < 1.0


def test_path_length_flat():
    g = Grid(np.zeros((3, 10)), 2.0)
    path = [(1, c) for c in range(10)]
    assert path_length_3d(g, path) == 9 * 2.0


def test_hazard_exposure_counts_steep_edges():
    g = inclined_plane(10, 10, cellsize=2.0, slope_deg=30.0)
    path = [(r, 0) for r in range(10)]  # straight up the 30-deg slope
    assert hazard_exposure(g, path, max_slope_deg=25.0) == 9
    assert hazard_exposure(g, path, max_slope_deg=35.0) == 0


def test_frechet_identical_and_offset():
    g = Grid(np.zeros((3, 10)), 2.0)
    p = [(0, c) for c in range(10)]
    q = [(1, c) for c in range(10)]
    assert discrete_frechet(g, p, p) == 0.0
    assert discrete_frechet(g, p, q) == 2.0  # one cell * cellsize


def test_stability_tiny_error_is_identical():
    g = inclined_plane(14, 14, cellsize=2.0, slope_deg=10.0)
    model = DEMErrorModel(sill_m2=1e-12, range_m=6.0)
    rep = route_stability(g, (0, 0), (13, 13), model, n=10, seed=0)
    assert rep.baseline_path is not None
    assert rep.n_feasible == 10
    assert rep.frac_identical == 1.0
    assert rep.mean_jaccard == 1.0


def test_stability_ranges_under_real_error():
    g = random_smooth_terrain(18, 18, seed=2, relief_m=4.0, passes=10)
    model = DEMErrorModel(sill_m2=1.5, range_m=6.0)
    rep = route_stability(g, (0, 0), (17, 17), model, n=20, seed=1)
    assert rep.n_feasible > 0
    assert 0.0 <= rep.mean_jaccard <= 1.0
    assert 0.0 <= rep.frac_identical <= 1.0
    assert rep.frequency.max() <= 1.0 + 1e-9


def test_ablation_detects_decision_change():
    g = ridge_world(40, 40, cellsize=2.0, ridge_height_m=2.0, ridge_half_width=1, gap_col=20)
    configs = {
        "strict": PlannerConfig(max_slope_deg=25.0),   # must detour via gap
        "lenient": PlannerConfig(max_slope_deg=89.0),  # may cross the ridge
        "strict_copy": PlannerConfig(max_slope_deg=25.0),
    }
    rep = ablation(g, (0, 0), (39, 0), configs, baseline="strict")
    changed = rep.changed()
    assert "lenient" in changed
    assert "strict_copy" not in changed
    assert "strict" not in changed
