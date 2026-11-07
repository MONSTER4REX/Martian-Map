"""C4 radiation screening: a documented scalar dose with a validity range.

A single surface dose-rate scalar with simple exponential shielding
attenuation. This is a SCREENING scalar only (MARSWALK_CONTEXT.md C4): it does
not model GCR vs SEP spectra, solar cycle, storms, or geometry. The default
rate is a PLACEHOLDER; an owner sets a cited value and the conditions it holds
for (solar phase, quiet vs event) before it informs decisions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RadiationScreen(BaseModel):
    surface_dose_rate_usv_per_h: float = Field(default=20.0, gt=0)  # PLACEHOLDER
    attenuation_length_g_cm2: float = Field(default=30.0, gt=0)  # PLACEHOLDER
    validity: str = (
        "PLACEHOLDER screening scalar; surface GCR-like background only, no SEP "
        "events, no spectra/solar-cycle. Set a cited rate + conditions before use."
    )

    def dose_usv(self, duration_h: float, shielding_g_cm2: float = 0.0) -> float:
        """Screening dose (uSv) over a duration, with exponential shielding."""
        if duration_h < 0 or shielding_g_cm2 < 0:
            raise ValueError("duration and shielding must be non-negative")
        import math

        atten = math.exp(-shielding_g_cm2 / self.attenuation_length_g_cm2)
        return self.surface_dose_rate_usv_per_h * duration_h * atten
