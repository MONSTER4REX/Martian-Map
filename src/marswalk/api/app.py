"""FastAPI app exposing the planner. OpenAPI schema is auto-generated.

Endpoints:
  GET  /health         -> service status + version
  POST /routes         -> three-route comparison (RouteComparison)
  POST /probe          -> per-cell layer values (ProbeResponse)

Run locally:  uvicorn marswalk.api.app:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

from marswalk import __version__
from marswalk.api.schemas import ProbeRequest, ProbeResponse, RouteRequest
from marswalk.api.service import plan_three_routes, probe
from marswalk.planner.routes import RouteComparison


def create_app():
    from fastapi import FastAPI  # imported lazily so the package imports without fastapi
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Marswalk Planner API",
        version=__version__,
        description="EVA route planning: three-route comparison and a probe tool.",
    )
    # Permissive CORS for the local demo frontend; tighten before any deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/routes", response_model=RouteComparison)
    def routes(req: RouteRequest) -> RouteComparison:
        return plan_three_routes(req)

    @app.post("/probe", response_model=ProbeResponse)
    def probe_endpoint(req: ProbeRequest) -> ProbeResponse:
        return probe(req)

    return app


def export_openapi(path: str | Path) -> None:
    """Write the OpenAPI schema to a JSON file (for the frontend / docs)."""
    app = create_app()
    Path(path).write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")


try:  # a module-level app for `uvicorn marswalk.api.app:app`
    app = create_app()
except ImportError:  # pragma: no cover - fastapi missing
    app = None
