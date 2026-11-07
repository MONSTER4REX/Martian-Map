import networkx as nx
import numpy as np
import pytest

from marswalk.physics.locomotion import LocomotionParams, edge_time_s
from marswalk.planner.cost import distance_3d, horizontal_distance
from marswalk.planner.multiobjective import (
    _dominates,
    corridor_mask,
    plan_pareto,
)
from marswalk.planner.routes import PlannerConfig, _make_neighbors
from marswalk.planner.slope import edge_along_slope_deg, terrain_gradient
from marswalk.terrain.synthetic import random_smooth_terrain


def _nx_time_graph(grid, cfg, loco):
    g_col, g_row = terrain_gradient(grid)
    neighbors = _make_neighbors(grid, g_col, g_row, cfg, "full_constraint")
    G = nx.DiGraph()
    cs = grid.cellsize
    for r in range(grid.nrows):
        for c in range(grid.ncols):
            if not grid.is_known((r, c)):
                continue
            a = (r, c)
            for b, _cost in neighbors(a):
                dz = float(grid.elevation[b] - grid.elevation[a])
                horiz = horizontal_distance(b[0] - a[0], b[1] - a[1], cs)
                G.add_edge(a, b, t=edge_time_s(dz, horiz, loco),
                           d=distance_3d(dz, horiz))
    return G


def test_dominance_basics():
    assert _dominates((1.0, 2.0), (2.0, 3.0))
    assert _dominates((1.0, 2.0), (1.0, 3.0))
    assert not _dominates((1.0, 2.0), (1.0, 2.0))  # equal, not dominated
    assert not _dominates((1.0, 3.0), (2.0, 2.0))  # incomparable


def test_front_is_nondominated_and_nonempty():
    grid = random_smooth_terrain(14, 14, seed=3, relief_m=4.0, passes=10)
    front = plan_pareto(grid, (0, 0), (13, 13))
    assert len(front.routes) >= 1
    vecs = [tuple(r.cost[o] for o in front.objectives) for r in front.routes]
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j:
                assert not _dominates(a, b), f"{a} dominates {b}"


@pytest.mark.parametrize("seed", range(5))
def test_pareto_extremes_match_single_objective(seed):
    cfg = PlannerConfig(connectivity=8, max_slope_deg=25.0, max_cross_slope_deg=22.0)
    loco = LocomotionParams()
    grid = random_smooth_terrain(12, 12, seed=seed, relief_m=4.0, passes=10)
    start, goal = (0, 0), (11, 11)

    front = plan_pareto(grid, start, goal, objectives=("distance", "time"), pcfg=cfg, loco=loco)
    if not front.routes:
        pytest.skip("no route")

    # Per-objective extremes == single-objective Dijkstra on the SAME
    # full-constraint graph (independent oracle, NetworkX).
    G = _nx_time_graph(grid, cfg, loco)
    d_opt = nx.dijkstra_path_length(G, start, goal, weight="d")
    t_opt = nx.dijkstra_path_length(G, start, goal, weight="t")
    assert front.best_for("distance").cost["distance"] == pytest.approx(d_opt, rel=1e-9, abs=1e-9)
    assert front.best_for("time").cost["time"] == pytest.approx(t_opt, rel=1e-9, abs=1e-9)


def test_corridor_mask_restricts_cells():
    grid = random_smooth_terrain(20, 20, seed=1, relief_m=4.0, passes=10)
    start, goal = (0, 0), (19, 19)
    mask = corridor_mask(grid, start, goal, half_width_cells=2)
    assert mask[start] and mask[goal]
    assert not mask[0, 19]  # a far corner is outside the diagonal corridor
    front = plan_pareto(grid, start, goal, corridor=mask)
    for route in front.routes:
        for cell in route.path:
            assert mask[cell]
