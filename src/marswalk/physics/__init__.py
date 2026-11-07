"""P4 EVA physics: locomotion, consumables, walk-back.

Baseline, offline. ALL numeric parameters are PLACEHOLDERS with a stated source
of "none yet" — P4 owns replacing them with cited suit/rover values and a
validity range (see docs/physics.md). Mars surface gravity 3.71 m/s^2 is the
only hard physical constant used here.

This module is the GATE for multi-objective search (MARSWALK_CONTEXT.md §3):
walk-back and consumables must pass tests before NAMOA*/orienteering begins.
"""

from marswalk.physics.consumables import (
    BASELINE_CONSUMABLES,
    CONSERVATIVE_CONSUMABLES,
    Consumable,
    ConsumablesModel,
)
from marswalk.physics.locomotion import (
    LocomotionParams,
    activity_factor,
    edge_energy_j,
    edge_time_s,
    speed_ms,
)
from marswalk.physics.radiation import RadiationScreen
from marswalk.physics.slip import SlipParams, slip_proxy
from marswalk.physics.thermal import ThermalParams, surface_temp_range_k
from marswalk.physics.walkback import (
    WalkbackReport,
    WalkbackScenario,
    check_walkback,
    standard_scenarios,
    time_to_haven_field,
)

__all__ = [
    "LocomotionParams",
    "speed_ms",
    "edge_time_s",
    "edge_energy_j",
    "activity_factor",
    "Consumable",
    "ConsumablesModel",
    "BASELINE_CONSUMABLES",
    "CONSERVATIVE_CONSUMABLES",
    "WalkbackScenario",
    "WalkbackReport",
    "time_to_haven_field",
    "check_walkback",
    "standard_scenarios",
    "ThermalParams",
    "surface_temp_range_k",
    "RadiationScreen",
    "SlipParams",
    "slip_proxy",
]
