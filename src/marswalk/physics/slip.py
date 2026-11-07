"""C6 slip proxy: a monotone slope->slip-risk scalar (PLACEHOLDER).

A dimensionless slip-risk proxy in [0, 1) that grows with slope, as an
informational layer / soft objective. It is NOT a validated terramechanics
model and NOT a failure probability (never present an unvalidated risk score as
a probability, MARSWALK_CONTEXT.md §3). Negative rover cases (Spirit's Troy,
Opportunity's Purgatory, Curiosity ripple slips) are the validation targets.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class SlipParams(BaseModel):
    char_slope_deg: float = Field(default=20.0, gt=0)  # slope at which risk ~63%
    sharpness: float = Field(default=2.0, gt=0)  # curve steepness
    validity: str = (
        "PLACEHOLDER proxy, not a probability; validate against negative rover "
        "cases (Troy, Purgatory, ripple slips) before use."
    )


def slip_proxy(slope_deg: float, p: SlipParams | None = None) -> float:
    """Slip-risk proxy in [0, 1): 0 on flat ground, increasing with slope."""
    p = p or SlipParams()
    s = abs(slope_deg) / p.char_slope_deg
    return 1.0 - math.exp(-(s**p.sharpness))
