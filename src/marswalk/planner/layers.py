"""Layer classification: every layer is hard / objective / informational.

Core rule (MARSWALK_CONTEXT.md §3): each layer must declare how it enters the
plan. Missing data is unknown, never "safe".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LayerKind(str, Enum):
    HARD = "hard"  # can veto an edge (exclusion)
    OBJECTIVE = "objective"  # contributes to edge cost
    INFORMATIONAL = "informational"  # shown, never changes the route


@dataclass(frozen=True)
class LayerSpec:
    name: str
    kind: LayerKind
    units: str
    note: str


BASELINE_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec(
        "elevation",
        LayerKind.INFORMATIONAL,
        "m",
        "Base DEM; source of all derivatives. NaN = unknown.",
    ),
    LayerSpec(
        "unknown_mask",
        LayerKind.HARD,
        "bool",
        "NaN/nodata cells are never traversed (A4).",
    ),
    LayerSpec(
        "along_slope",
        LayerKind.HARD,
        "deg",
        "Along-path slope: hard exclusion above max_slope_deg AND an objective "
        "cost via the slope penalty (A7, A8).",
    ),
    LayerSpec(
        "cross_slope",
        LayerKind.HARD,
        "deg",
        "Side-hill angle: hard exclusion above max_cross_slope_deg (A9).",
    ),
)
