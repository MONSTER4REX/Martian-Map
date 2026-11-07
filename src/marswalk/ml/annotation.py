"""Rock annotation schema.

Boxes are in tile pixel coordinates (top-left origin). A tile carries its map
origin and pixel size so a box's real diameter is known. SAM/model labels are
PROPOSALS, not ground truth — the `source` field records that distinction, and
a spatial holdout (ml.splits) keeps train/test disjoint.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class LabelSource(str, Enum):
    MANUAL = "manual"  # human-drawn ground truth
    SAM_PROPOSAL = "sam_proposal"  # Segment Anything proposal (NOT ground truth)
    MODEL_PRED = "model_pred"  # a detector's prediction


class BBox(BaseModel):
    x: float  # pixels, top-left
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    def area(self) -> float:
        return self.w * self.h


class RockLabel(BaseModel):
    bbox: BBox
    source: LabelSource = LabelSource.MANUAL
    confidence: float = 1.0

    def diameter_m(self, px_size_m: float) -> float:
        return max(self.bbox.w, self.bbox.h) * px_size_m


class Tile(BaseModel):
    id: str
    path: str = ""  # image path (not required for schema tests)
    origin_x_m: float = 0.0  # map x of the tile's top-left
    origin_y_m: float = 0.0
    px_size_m: float = Field(default=0.25, gt=0)
    width: int = 256
    height: int = 256
    labels: list[RockLabel] = Field(default_factory=list)

    def center_xy_m(self) -> tuple[float, float]:
        return (
            self.origin_x_m + 0.5 * self.width * self.px_size_m,
            self.origin_y_m + 0.5 * self.height * self.px_size_m,
        )


class AnnotationSet(BaseModel):
    tiles: list[Tile] = Field(default_factory=list)

    def by_id(self, tile_id: str) -> Tile:
        for t in self.tiles:
            if t.id == tile_id:
                return t
        raise KeyError(tile_id)

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tiles:
            for lab in t.labels:
                counts[lab.source.value] = counts.get(lab.source.value, 0) + 1
        return counts

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AnnotationSet:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
