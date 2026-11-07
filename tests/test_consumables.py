import pytest

from marswalk.physics.consumables import (
    BASELINE_CONSUMABLES,
    CONSERVATIVE_CONSUMABLES,
    Consumable,
    ConsumablesModel,
)


def test_binding_resource_sets_budget():
    m = ConsumablesModel(
        items=[
            Consumable(name="o2", capacity=100.0, rate_per_s=1.0),
            Consumable(name="water", capacity=300.0, rate_per_s=1.0),
        ]
    )
    assert m.time_budget_s() == pytest.approx(100.0)
    assert m.binding_resource() == "o2"


def test_activity_and_degradation_shrink_budget():
    m = BASELINE_CONSUMABLES
    base = m.time_budget_s()
    assert m.time_budget_s(activity_factor=2.0) == pytest.approx(base / 2.0)
    assert m.time_budget_s(degradation=2.0) == pytest.approx(base / 2.0)


def test_conservative_is_tighter_than_baseline():
    assert CONSERVATIVE_CONSUMABLES.time_budget_s() < BASELINE_CONSUMABLES.time_budget_s()


def test_empty_model_raises():
    with pytest.raises(ValueError):
        ConsumablesModel(items=[]).time_budget_s()
