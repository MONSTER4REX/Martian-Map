import numpy as np

from marswalk.planner.layers import BASELINE_LAYERS, LayerKind
from marswalk.planner.routes import PlannerConfig, compare_routes, plan_route
from marswalk.terrain.synthetic import inclined_plane, punch_nodata, ridge_world


def test_layer_classification_declared():
    kinds = {layer.name: layer.kind for layer in BASELINE_LAYERS}
    assert kinds["elevation"] == LayerKind.INFORMATIONAL
    assert kinds["unknown_mask"] == LayerKind.HARD
    assert kinds["along_slope"] == LayerKind.HARD
    assert kinds["cross_slope"] == LayerKind.HARD


def test_ridge_decision_flip():
    # A single steep ridge row with one flat gap. Straight-line crossing is
    # cheap in distance but violates the slope limit; the mission route must
    # detour through the gap.
    grid = ridge_world(nrows=40, ncols=40, cellsize=2.0, ridge_height_m=2.0,
                        ridge_half_width=1, gap_col=20)
    cmp = compare_routes(grid, start=(0, 0), goal=(39, 0))

    shortest = cmp.by_name("Shortest (3-D distance)")
    mission = cmp.by_name("Full-constraint (mission)")

    assert shortest.feasible and mission.feasible
    # Shortest takes the steep shortcut and is NOT mission-feasible.
    assert not shortest.mission_feasible
    assert shortest.metrics.n_hard_violations > 0
    assert "REJECTED" in shortest.reason
    # Mission route satisfies the hard exclusions and is longer (it detours).
    assert mission.mission_feasible
    assert mission.metrics.n_hard_violations == 0
    assert mission.metrics.length_3d_m > shortest.metrics.length_3d_m
    assert "NOT mission-feasible" in cmp.summary


def test_unknown_start_or_goal_rejected():
    grid = inclined_plane(10, 10, slope_deg=5.0)
    grid = punch_nodata(grid, [(0, 0)])
    cmp = compare_routes(grid, start=(0, 0), goal=(9, 9))
    assert all(not r.feasible for r in cmp.routes)
    assert "unknown/nodata" in cmp.summary


def test_out_of_bounds_rejected():
    grid = inclined_plane(10, 10, slope_deg=5.0)
    cmp = compare_routes(grid, start=(0, 0), goal=(99, 99))
    assert all(not r.feasible for r in cmp.routes)
    assert "out of bounds" in cmp.summary


def test_nodata_wall_blocks_shortest():
    # Wall off the goal corner with a full nodata column; no route can reach it.
    grid = inclined_plane(8, 8, slope_deg=3.0)
    wall = [(r, 4) for r in range(8)]
    grid = punch_nodata(grid, wall)
    cmp = compare_routes(grid, start=(0, 0), goal=(7, 7))
    for r in cmp.routes:
        assert not r.feasible
        assert "unreachable" in r.reason


def test_full_constraint_infeasible_but_shortest_ok():
    # Steep plane + a slope limit below the gentlest row-changing edge: the
    # mission route cannot change rows, so a cross-row goal is unreachable
    # under hard constraints while the shortest route still gets there.
    grid = inclined_plane(10, 10, cellsize=2.0, slope_deg=10.0)
    cfg = PlannerConfig(connectivity=8, max_slope_deg=5.0, max_cross_slope_deg=89.0)
    cmp = compare_routes(grid, start=(0, 0), goal=(9, 0), config=cfg)

    shortest = cmp.by_name("Shortest (3-D distance)")
    mission = cmp.by_name("Full-constraint (mission)")
    assert shortest.feasible
    assert not mission.feasible
    assert "unreachable under hard exclusions" in mission.reason


def test_determinism():
    grid = ridge_world(30, 30)
    p1, c1 = plan_route(grid, (0, 0), (29, 0), "full_constraint")
    p2, c2 = plan_route(grid, (0, 0), (29, 0), "full_constraint")
    assert p1 == p2
    assert c1 == c2


def test_serialization_roundtrip():
    grid = ridge_world(20, 20, ridge_height_m=2.0, ridge_half_width=1, gap_col=10)
    cmp = compare_routes(grid, (0, 0), (19, 0))
    blob = cmp.model_dump_json()
    assert '"start"' in blob and '"summary"' in blob
    assert isinstance(np.array(cmp.by_name("Full-constraint (mission)").path), np.ndarray)
