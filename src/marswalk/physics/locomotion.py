"""Locomotion: walking speed, traverse time, and a transparent energy model.

Units: distances meters, times seconds, speeds m/s, energy joules, mass kg,
slope degrees. Mars surface gravity g = 3.71 m/s^2 (the only cited constant).

EVERYTHING ELSE IS A PLACEHOLDER. The speed-vs-slope curve and the energy
coefficients are simple, monotone, documented stand-ins — NOT validated suited-
EVA biomechanics. P4 must replace them with cited values and a validity range.
Downhill is treated the same as uphill (|slope|), which is a conservative
simplification for walk-back timing; revisit when a real model lands.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

GRAVITY_MARS_MS2 = 3.71  # cited physical constant


class LocomotionParams(BaseModel):
    """Placeholder locomotion parameters (see module docstring)."""

    level_speed_ms: float = Field(default=1.0, gt=0)  # suited walk on flat ground
    min_speed_ms: float = Field(default=0.15, gt=0)  # floor on steep ground
    slope_speed_falloff_per_deg: float = Field(default=0.03, ge=0)  # speed loss/deg
    activity_slope_coeff: float = Field(default=1.0, ge=0)  # metabolic/consumable uplift

    # Energy model (placeholder).
    total_mass_kg: float = Field(default=180.0, gt=0)  # crew + suit + pack
    cot_level: float = Field(default=0.20, ge=0)  # dimensionless cost of transport, flat
    muscle_efficiency: float = Field(default=0.25, gt=0, le=1)  # climb work efficiency

    gravity_ms2: float = Field(default=GRAVITY_MARS_MS2, gt=0)


def speed_ms(along_slope_deg: float, p: LocomotionParams) -> float:
    """Walking speed on a given along-path slope. Monotone decreasing, floored."""
    s = p.level_speed_ms * (1.0 - p.slope_speed_falloff_per_deg * abs(along_slope_deg))
    return max(s, p.min_speed_ms)


def edge_time_s(dz: float, horizontal: float, p: LocomotionParams) -> float:
    """Traverse time for an edge from its vertical and horizontal extents."""
    dist = math.hypot(dz, horizontal)
    along = math.degrees(math.atan2(abs(dz), horizontal))
    return dist / speed_ms(along, p)


def activity_factor(along_slope_deg: float, p: LocomotionParams) -> float:
    """Metabolic/consumable uplift on slopes (>= 1). Placeholder shape."""
    return 1.0 + p.activity_slope_coeff * abs(along_slope_deg) / 10.0


def edge_energy_j(dz: float, horizontal: float, p: LocomotionParams) -> float:
    """Metabolic energy for an edge (joules). Placeholder model.

    Level term: cost-of-transport work ``cot_level * m * g * horizontal``.
    Climb term: gravitational work for positive elevation gain divided by
    muscle efficiency; no energy credit for descent (conservative).
    """
    level = p.cot_level * p.total_mass_kg * p.gravity_ms2 * horizontal
    climb = p.total_mass_kg * p.gravity_ms2 * max(dz, 0.0) / p.muscle_efficiency
    return level + climb
