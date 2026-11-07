"""P1 data/GIS layer (offline parts).

Derivatives and QA run on any in-memory DEM (synthetic here; real COG windows
via `io` once an owner wires rasterio). Everything is NaN-aware: unknown stays
unknown, never silently filled.
"""

from marswalk.data.derivatives import aspect_degrees, curvature, hillshade, slope_degrees
from marswalk.data.qa import DTMQAReport, run_dtm_qa

__all__ = [
    "slope_degrees",
    "aspect_degrees",
    "hillshade",
    "curvature",
    "DTMQAReport",
    "run_dtm_qa",
]
