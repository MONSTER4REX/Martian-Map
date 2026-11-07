"""P2 rock detector (gated by the baseline planner).

The schema, spatial splits and detection metrics are pure and tested offline.
The training script (`ml.train`) needs torch + a GPU and is UNTESTED here — an
owner runs it on the RTX 4050 laptop and pastes back peak VRAM / F1 by size bin.
Per the gates (MARSWALK_CONTEXT.md §3): no ML before the baseline planner passes
(it does), and the rock detector ships only if an ablation shows rock-aware
routes reduce rock exposure vs slope-only.
"""

from marswalk.ml.annotation import AnnotationSet, BBox, LabelSource, RockLabel, Tile
from marswalk.ml.metrics_det import iou, match_detections, precision_recall_by_size
from marswalk.ml.splits import spatial_block_split, splits_are_disjoint

__all__ = [
    "BBox",
    "LabelSource",
    "RockLabel",
    "Tile",
    "AnnotationSet",
    "spatial_block_split",
    "splits_are_disjoint",
    "iou",
    "match_detections",
    "precision_recall_by_size",
]
