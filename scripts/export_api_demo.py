#!/usr/bin/env python
"""Export a static demo bundle + OpenAPI schema for the web frontend.

Writes:
  web/sample_routes.json  - {elevation, cellsize, start, goal, routes, summary}
  web/openapi.json        - the API schema

so web/index.html renders offline without a running server. Run:
    python scripts/export_api_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from marswalk.api.app import export_openapi
from marswalk.api.schemas import RouteRequest, TerrainSpec
from marswalk.api.service import build_grid, plan_three_routes

WEB = Path(__file__).resolve().parent.parent / "web"


def main() -> None:
    WEB.mkdir(exist_ok=True)
    terrain = TerrainSpec(kind="ridge", nrows=40, ncols=40, cellsize=2.0,
                          ridge_height_m=2.0, ridge_half_width=1, gap_col=20)
    start, goal = (0, 0), (39, 0)
    grid = build_grid(terrain)
    cmp = plan_three_routes(RouteRequest(terrain=terrain, start=start, goal=goal))

    elev = [[None if not np.isfinite(v) else round(float(v), 3) for v in row]
            for row in grid.elevation]
    bundle = {
        "elevation": elev,
        "cellsize": grid.cellsize,
        "start": list(start),
        "goal": list(goal),
        "routes": json.loads(cmp.model_dump_json())["routes"],
        "summary": cmp.summary,
    }
    (WEB / "sample_routes.json").write_text(json.dumps(bundle), encoding="utf-8")
    export_openapi(WEB / "openapi.json")
    print(f"wrote {WEB/'sample_routes.json'} and {WEB/'openapi.json'}")


if __name__ == "__main__":
    main()
