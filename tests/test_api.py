import importlib.util

import pytest

from marswalk.api.schemas import ProbeRequest, RouteRequest, TerrainSpec
from marswalk.api.service import build_grid, plan_three_routes, probe

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


def test_plan_three_routes_service():
    req = RouteRequest(
        terrain=TerrainSpec(kind="ridge", nrows=40, ncols=40, ridge_height_m=2.0,
                            ridge_half_width=1, gap_col=20),
        start=(0, 0),
        goal=(39, 0),
    )
    cmp = plan_three_routes(req)
    names = {r.name for r in cmp.routes}
    assert len(cmp.routes) == 3
    shortest = cmp.by_name("Shortest (3-D distance)")
    mission = cmp.by_name("Full-constraint (mission)")
    assert not shortest.mission_feasible
    assert mission.mission_feasible
    assert "Shortest (3-D distance)" in names


def test_probe_service():
    terrain = TerrainSpec(kind="random", nrows=20, ncols=20, seed=1, relief_m=5.0)
    r = probe(ProbeRequest(terrain=terrain, cell=(10, 10)))
    assert r.in_bounds and r.known
    assert r.elevation_m is not None and r.slope_deg is not None
    out = probe(ProbeRequest(terrain=terrain, cell=(99, 99)))
    assert not out.in_bounds and out.elevation_m is None


def test_inline_terrain_with_nodata():
    spec = TerrainSpec(kind="inline", cellsize=1.0, elevation=[[0.0, None], [1.0, 2.0]])
    g = build_grid(spec)
    assert g.shape == (2, 2)
    import numpy as np
    assert np.isnan(g.elevation[0, 1])


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
def test_fastapi_endpoints():
    from fastapi.testclient import TestClient

    from marswalk.api.app import create_app

    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ok"

    body = {
        "terrain": {"kind": "ridge", "nrows": 30, "ncols": 30,
                    "ridge_height_m": 2.0, "ridge_half_width": 1, "gap_col": 15},
        "start": [0, 0],
        "goal": [29, 0],
    }
    resp = client.post("/routes", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["routes"]) == 3
    assert "summary" in data

    pr = client.post("/probe", json={"terrain": body["terrain"], "cell": [10, 10]})
    assert pr.status_code == 200
    assert "elevation_m" in pr.json()


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
def test_openapi_has_paths():
    from marswalk.api.app import create_app

    schema = create_app().openapi()
    assert "/routes" in schema["paths"]
    assert "/probe" in schema["paths"]
