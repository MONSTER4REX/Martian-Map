"""P3 planner baseline (offline, synthetic terrain).

Public surface:
    Grid                     - 2.5-D elevation grid (NaN = unknown = blocked)
    neighbor_offsets         - 8/16 connectivity offsets
    astar                    - generic A* over a lazily-defined graph
    PlannerConfig            - thresholds and policy knobs
    RouteResult, RouteComparison, compare_routes - the three-route output
    LayerKind, BASELINE_LAYERS - layer classification (hard/objective/informational)
"""

from marswalk.planner.astar import astar
from marswalk.planner.connectivity import neighbor_offsets
from marswalk.planner.grid import Grid
from marswalk.planner.layers import BASELINE_LAYERS, LayerKind, LayerSpec
from marswalk.planner.routes import (
    PlannerConfig,
    RouteComparison,
    RouteResult,
    compare_routes,
    plan_route,
)

# NOTE: the physics-bridging planners live in their own modules to keep this
# package free of a physics dependency (avoids an import cycle, since physics
# imports planner primitives). Import them directly:
#   from marswalk.planner.mission import plan_walkback_route, ...
#   from marswalk.planner.multiobjective import plan_pareto, ...

__all__ = [
    "Grid",
    "neighbor_offsets",
    "astar",
    "PlannerConfig",
    "RouteResult",
    "RouteComparison",
    "compare_routes",
    "plan_route",
    "LayerKind",
    "LayerSpec",
    "BASELINE_LAYERS",
]
