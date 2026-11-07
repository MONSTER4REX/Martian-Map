"""Consumables: per-resource capacity and rate -> a time budget.

Units: capacity in a resource's own unit, rate in unit/second, budget seconds.
The binding (shortest-lasting) resource sets the EVA time budget.

ALL VALUES ARE PLACEHOLDERS. The baseline and conservative sets below are
round, defensible-looking stand-ins so the walk-back math can run — they are
NOT sourced from a suit spec. P4 must replace them with cited life-support
numbers and a validity range. "units" are intentionally abstract where a real
figure is not known.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Consumable(BaseModel):
    name: str
    capacity: float = Field(gt=0)  # total available, resource's own unit
    rate_per_s: float = Field(gt=0)  # nominal consumption at activity_factor = 1
    unit: str = "unit"

    def duration_s(self, activity_factor: float = 1.0) -> float:
        """How long this resource lasts at the given activity uplift (>= 1)."""
        return self.capacity / (self.rate_per_s * max(activity_factor, 1e-9))


class ConsumablesModel(BaseModel):
    """A set of consumables; the EVA time budget is the binding resource."""

    items: list[Consumable]

    def time_budget_s(self, activity_factor: float = 1.0, degradation: float = 1.0) -> float:
        """Min duration across resources.

        ``degradation`` >= 1 scales all consumption rates up (e.g. a leaking
        suit), shrinking the budget proportionally.
        """
        if not self.items:
            raise ValueError("ConsumablesModel has no items")
        return min(c.duration_s(activity_factor * degradation) for c in self.items)

    def binding_resource(self, activity_factor: float = 1.0, degradation: float = 1.0) -> str:
        return min(
            self.items, key=lambda c: c.duration_s(activity_factor * degradation)
        ).name


# A ~6 hour-ish nominal EVA budget from the binding resource. PLACEHOLDERS.
BASELINE_CONSUMABLES = ConsumablesModel(
    items=[
        Consumable(name="oxygen", capacity=21600.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="water", capacity=25200.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="battery", capacity=28800.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="co2_scrub", capacity=23400.0, rate_per_s=1.0, unit="abstract-s"),
    ]
)

# Conservative: smaller margins -> shorter, binding budget. PLACEHOLDERS.
CONSERVATIVE_CONSUMABLES = ConsumablesModel(
    items=[
        Consumable(name="oxygen", capacity=16200.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="water", capacity=18000.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="battery", capacity=21600.0, rate_per_s=1.0, unit="abstract-s"),
        Consumable(name="co2_scrub", capacity=16200.0, rate_per_s=1.0, unit="abstract-s"),
    ]
)
