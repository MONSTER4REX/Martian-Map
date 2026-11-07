"""C3 thermal screening (PLACEHOLDER model).

A transparent, deliberately simple surface-temperature screening estimate so
EVA-time windows can be explored. It is NOT a validated thermal model: ground
skin temperature and air temperature are not interchangeable, and this must be
validated against a MATCHED quantity (e.g. MEDA ground-temperature) before it
informs decisions (MARSWALK_CONTEXT.md C3). Treat outputs as order-of-magnitude
screening within the stated validity range only.

Constants: Stefan-Boltzmann sigma, and the mean Mars solar irradiance ~586 W/m^2
(a real orbital value, used as a scale). Everything else is a placeholder.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SIGMA = 5.670374419e-8  # W m^-2 K^-4
MARS_MEAN_IRRADIANCE_WM2 = 586.0  # mean top-of-atmosphere at Mars


class ThermalParams(BaseModel):
    albedo: float = Field(default=0.25, ge=0, lt=1)
    emissivity: float = Field(default=0.95, gt=0, le=1)
    # diurnal damping: higher thermal inertia -> smaller day/night swing
    ti_reference: float = Field(default=200.0, gt=0)  # J m^-2 K^-1 s^-1/2 scale
    night_floor_k: float = Field(default=150.0, gt=0)  # placeholder night asymptote
    validity: str = (
        "PLACEHOLDER screening only; clear-sky, flat, no slope/azimuth; validate "
        "vs MEDA ground temperature before use."
    )


def radiative_peak_k(insolation_wm2: float, p: ThermalParams) -> float:
    """Radiative-equilibrium peak skin temperature (K). Upper-bound-ish."""
    absorbed = (1.0 - p.albedo) * max(insolation_wm2, 0.0)
    return (absorbed / (p.emissivity * SIGMA)) ** 0.25 if absorbed > 0 else p.night_floor_k


def surface_temp_range_k(
    thermal_inertia: float, noon_insolation_wm2: float, p: ThermalParams | None = None
) -> tuple[float, float]:
    """(T_min, T_max) skin temperature over a diurnal cycle (K), screening only.

    Peak from radiative equilibrium; the day/night amplitude shrinks with
    thermal inertia (damping), so high-inertia ground has a smaller swing.
    """
    p = p or ThermalParams()
    tmax = radiative_peak_k(noon_insolation_wm2, p)
    damping = 1.0 + max(thermal_inertia, 0.0) / p.ti_reference
    amplitude = (tmax - p.night_floor_k) / damping
    tmin = tmax - amplitude
    return tmin, tmax
