"""Data manifest: a STAC-ish record of every layer, its provenance and license.

Encodes the spec requirement that each layer declares native resolution, an
effective-resolution tag, uncertainty, units, its hard/objective/informational
class, and its license + verification status (so coarse layers are never shown
as EVA-scale detail, and share-alike / non-commercial terms are tracked).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from marswalk.planner.layers import LayerKind


class VerifyStatus(str, Enum):
    UNVERIFIED = "unverified"  # from the planning brief, not confirmed at source
    VERIFIED = "verified"  # an owner confirmed license + provenance firsthand


class LayerRecord(BaseModel):
    id: str
    title: str
    kind: LayerKind
    product_type: str  # e.g. "DEM", "orthoimage", "thermal_inertia"
    source_url: str
    crs: str = "IAU:49900 (Mars 2000)"  # placeholder; confirm per product
    native_res_m: float = Field(gt=0)
    effective_res_m: float = Field(gt=0)
    units: str
    license: str  # required; "UNKNOWN" is allowed but flagged by status
    uncertainty: str = "unspecified"
    status: VerifyStatus = VerifyStatus.UNVERIFIED
    notes: str = ""

    @model_validator(mode="after")
    def _check_resolution(self) -> LayerRecord:
        if self.effective_res_m < self.native_res_m - 1e-9:
            raise ValueError(
                f"effective_res_m ({self.effective_res_m}) finer than native "
                f"({self.native_res_m}) for layer {self.id!r}: never upsample "
                "coarse data into false detail"
            )
        return self


class Manifest(BaseModel):
    aoi: str = "Jezero"
    crs: str = "IAU:49900 (Mars 2000)"
    layers: list[LayerRecord] = Field(default_factory=list)

    def by_kind(self, kind: LayerKind) -> list[LayerRecord]:
        return [layer for layer in self.layers if layer.kind == kind]

    def unverified(self) -> list[str]:
        return [layer.id for layer in self.layers if layer.status == VerifyStatus.UNVERIFIED]

    def coarser_than(self, res_m: float) -> list[str]:
        """Layers whose effective resolution is coarser than res_m (not EVA-scale)."""
        return [layer.id for layer in self.layers if layer.effective_res_m > res_m]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
