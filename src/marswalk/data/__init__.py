"""P1 data/GIS layer (offline parts).

Derivatives and QA run on any in-memory DEM (synthetic here; real COG windows
via `io` once an owner wires rasterio). Everything is NaN-aware: unknown stays
unknown, never silently filled.
"""

from marswalk.data.derivatives import aspect_degrees, curvature, hillshade, slope_degrees
from marswalk.data.error_model import (
    DEMErrorModel,
    empirical_variogram,
    ensemble,
    fit_exponential_variogram,
    perturbed_grid,
    sample_error,
)
from marswalk.data.io import WindowSpec, array_to_grid, bounds_to_window, read_cog_window
from marswalk.data.manifest import LayerRecord, Manifest, VerifyStatus
from marswalk.data.qa import DTMQAReport, run_dtm_qa

__all__ = [
    "slope_degrees",
    "aspect_degrees",
    "hillshade",
    "curvature",
    "DTMQAReport",
    "run_dtm_qa",
    "DEMErrorModel",
    "sample_error",
    "perturbed_grid",
    "ensemble",
    "empirical_variogram",
    "fit_exponential_variogram",
    "Manifest",
    "LayerRecord",
    "VerifyStatus",
    "WindowSpec",
    "bounds_to_window",
    "array_to_grid",
    "read_cog_window",
]
