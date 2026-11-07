"""C7 rock size-frequency distribution (SFD) fit.

Baseline model: an exponential cumulative *number* density,
    N(>=D) / area = N0 * exp(-lambda * D),
fit by linear regression of log density vs diameter. This is a generic,
checkable fit — NOT the site-calibrated Golombek & Rapp cumulative-fractional-
*area* model CFA(D)=k·exp(-q(k)·D), which is the literature standard for Mars
rock hazard; an owner should calibrate that against real counts (and the
pretrained rock-abundance prior) before it drives routing. Units: diameters m,
area m^2, density rocks/m^2, lambda 1/m.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class SFDFit(BaseModel):
    model: str
    n0_per_m2: float  # intercept density at D=0
    lambda_per_m: float  # decay rate
    r2: float
    n_rocks: int
    area_m2: float


def fit_exponential_sfd(diameters_m, area_m2: float) -> SFDFit:
    """Fit N(>=D)/area = N0 exp(-lambda D) by log-linear regression."""
    d = np.sort(np.asarray(diameters_m, dtype=float))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 2:
        raise ValueError("need at least 2 positive diameters")
    if not (area_m2 > 0):
        raise ValueError("area_m2 must be positive")

    uniq = np.unique(d)
    # cumulative count with diameter >= value
    counts = np.array([int(np.sum(d >= v)) for v in uniq], dtype=float)
    density = counts / area_m2
    y = np.log(density)
    A = np.column_stack([np.ones_like(uniq), uniq])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return SFDFit(
        model="exponential_cumulative_number",
        n0_per_m2=float(np.exp(intercept)),
        lambda_per_m=float(-slope),
        r2=r2,
        n_rocks=int(d.size),
        area_m2=float(area_m2),
    )


def cumulative_number_density(fit: SFDFit, diameter_m: float) -> float:
    """Predicted N(>=D) per m^2."""
    return fit.n0_per_m2 * float(np.exp(-fit.lambda_per_m * diameter_m))


def expected_count_at_least(fit: SFDFit, diameter_m: float, area_m2: float) -> float:
    """Expected number of rocks >= diameter over an area."""
    return cumulative_number_density(fit, diameter_m) * area_m2
