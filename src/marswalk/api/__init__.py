"""P3/P5 service API (co-owned OpenAPI schema).

`service.py` is a pure layer (pydantic in/out, no web framework) and is fully
tested offline. `app.py` wires it into FastAPI. Terrain is synthetic/demo here
(the sandbox has no real DEM); a COG-backed terrain source plugs in later via
`data.io`.
"""

from marswalk.api.schemas import ProbeRequest, ProbeResponse, RouteRequest, TerrainSpec
from marswalk.api.service import plan_three_routes, probe

__all__ = [
    "TerrainSpec",
    "RouteRequest",
    "ProbeRequest",
    "ProbeResponse",
    "plan_three_routes",
    "probe",
]
