# P3/P5 API + web demo

Status: **complete and tested offline** (FastAPI + a static React map). Owners:
P3 + P5 (schema co-owned). Terrain is synthetic/demo in this build; a COG-backed
source plugs in via `data.io` later.

## Service layer (`api/service.py`) — pure, tested

- `build_grid(TerrainSpec)` — demo terrain (`ridge` / `random` / `inline` with
  null = nodata).
- `plan_three_routes(RouteRequest)` → `RouteComparison` (the three routes).
- `probe(ProbeRequest)` → `ProbeResponse` (elevation, slope, aspect, hillshade,
  known/in-bounds) for a cell.

## FastAPI app (`api/app.py`)

- `GET /health`, `POST /routes`, `POST /probe`; OpenAPI auto-generated.
- `export_openapi(path)` dumps the schema.
- Run: `pip install -e ".[api]"` then `uvicorn marswalk.api.app:app --reload`.
  Docs at `/docs`, schema at `/openapi.json`.
- Tested with FastAPI's TestClient (`tests/test_api.py`).

## Web demo (`web/`)

Static page (`web/index.html`, React UMD) showing the three-route comparison,
per-route metrics + reasons, route toggles, and a click-to-probe tool. It loads
`web/sample_routes.json` (exported bundle) so it renders **offline**.

```bash
python scripts/export_api_demo.py          # writes web/sample_routes.json + openapi.json
python -m http.server 8777 --directory web # then open http://127.0.0.1:8777/
```

To drive it from a live server instead, run the API and point the frontend's
fetch at `/routes` + `/probe` (same request bodies as the exported bundle).

Verified in-browser: on the ridge demo, shortest/slope-aware cross the ridge and
are rejected (2 hard-exclusion edges, 45°); the mission route detours through
the gap (+37.4%). CORS is wide open for the local demo — tighten before any
deployment.
