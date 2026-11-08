"""P6 validation: route metrics, route stability, ablation harness.

Offline. These turn the planner's claims into measurable, reportable numbers:
route-metric comparisons, how often the recommended route changes under DEM
perturbation (the headline uncertainty figure), and whether a feature actually
changes decisions (the ablation gate).
"""

from marswalk.validation.ablation import (
    AblationReport,
    RockAblationReport,
    ablation,
    rock_ablation,
)
from marswalk.validation.metrics import (
    discrete_frechet,
    hazard_exposure,
    jaccard,
    path_cells,
    path_length_3d,
)
from marswalk.validation.stability import StabilityReport, route_stability

__all__ = [
    "path_cells",
    "jaccard",
    "path_length_3d",
    "discrete_frechet",
    "hazard_exposure",
    "route_stability",
    "StabilityReport",
    "ablation",
    "AblationReport",
    "rock_ablation",
    "RockAblationReport",
]
