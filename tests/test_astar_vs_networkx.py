"""Validate A* optimality against an independent oracle (NetworkX Dijkstra).

Both the planner and the oracle are built from the SAME edge generator
(`_make_neighbors`), so this isolates the search: A* must return exactly the
optimal cost of the planner's graph, on randomized terrain, nodata, both
connectivities, and all three policies.
"""

import networkx as nx
import numpy as np
import pytest

from marswalk.planner.routes import POLICIES, PlannerConfig, _make_neighbors, plan_route
from marswalk.planner.slope import terrain_gradient
from marswalk.terrain.synthetic import punch_nodata, random_smooth_terrain


def _build_nx_graph(grid, cfg, policy):
    g_col, g_row = terrain_gradient(grid)
    neighbors = _make_neighbors(grid, g_col, g_row, cfg, policy)
    G = nx.DiGraph()
    for r in range(grid.nrows):
        for c in range(grid.ncols):
            if not grid.is_known((r, c)):
                continue
            node = (r, c)
            G.add_node(node)
            for nbr, w in neighbors(node):
                G.add_edge(node, nbr, weight=w)
    return G


def _path_weight(G, path):
    return sum(G[a][b]["weight"] for a, b in zip(path[:-1], path[1:], strict=True))


def _known_cells(grid):
    return [tuple(x) for x in np.argwhere(grid.known_mask)]


CASES = [
    (seed, size, conn, policy)
    for seed in range(6)
    for size in (8, 13)
    for conn in (8, 16)
    for policy in POLICIES
]


@pytest.mark.parametrize("seed,size,conn,policy", CASES)
def test_astar_matches_dijkstra(seed, size, conn, policy):
    rng = np.random.default_rng(seed)
    grid = random_smooth_terrain(size, size, cellsize=2.0, seed=seed, relief_m=15.0, passes=4)

    # Punch a few random nodata holes.
    n_holes = rng.integers(0, max(1, size * size // 10))
    holes = {
        (int(rng.integers(size)), int(rng.integers(size))) for _ in range(int(n_holes))
    }
    grid = punch_nodata(grid, list(holes))

    known = _known_cells(grid)
    if len(known) < 2:
        pytest.skip("not enough known cells")
    start = known[int(rng.integers(len(known)))]
    goal = known[int(rng.integers(len(known)))]
    if start == goal:
        goal = known[(known.index(goal) + 1) % len(known)]

    cfg = PlannerConfig(connectivity=conn, max_slope_deg=20.0, max_cross_slope_deg=18.0)
    G = _build_nx_graph(grid, cfg, policy)
    path, cost = plan_route(grid, start, goal, policy, cfg)

    if nx.has_path(G, start, goal):
        oracle = nx.dijkstra_path_length(G, start, goal, weight="weight")
        assert path is not None
        # A* cost equals Dijkstra's optimal cost.
        assert cost == pytest.approx(oracle, rel=1e-9, abs=1e-9)
        # Returned path is real and its summed weight equals the reported cost.
        assert path[0] == start and path[-1] == goal
        assert _path_weight(G, path) == pytest.approx(cost, rel=1e-9, abs=1e-9)
    else:
        assert path is None
        assert cost == float("inf")
