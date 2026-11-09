"""Guards for the web-demo exporter helpers (offline, fast).

The exporter ships the real engine's fields to the browser; these tests pin the
serialization contract (inf/NaN -> null, hillshade in range) and prove the
browser's survival arithmetic reproduces the engine on a tiny grid.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.survival import survival_envelope
from marswalk.physics.walkback import standard_scenarios
from marswalk.terrain.synthetic import random_smooth_terrain

ROOT = Path(__file__).resolve().parent.parent


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_jezero_demo", ROOT / "scripts" / "export_jezero_demo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


EXP = _load_exporter()


def test_field_to_json_maps_nonfinite_to_null():
    a = np.array([[0.0, np.inf], [np.nan, 123.456]])
    out = EXP.field_to_json(a)
    assert out == [[0.0, None], [None, 123.5]]


def test_hillshade_in_byte_range():
    grid = random_smooth_terrain(12, 12, cellsize=25.0, seed=1, relief_m=40.0, passes=6)
    hs = EXP.hillshade(grid.elevation, grid.cellsize)
    assert hs.shape == grid.shape
    assert int(hs.min()) >= 0 and int(hs.max()) <= 255


def test_browser_rule_matches_engine():
    """The page's margin = budget - reach - haven - reserve must agree with the
    engine's `survivable` mask cell-for-cell."""
    grid = random_smooth_terrain(18, 18, cellsize=25.0, seed=3, relief_m=50.0, passes=8)
    start, base, rover = (14, 10), (2, 2), (15, 9)
    scen = standard_scenarios(base, rover)
    budget = 3000.0
    reserve = 600.0
    consumables = ConsumablesModel(items=[Consumable(name="o2", capacity=budget, rate_per_s=1.0)])

    env = survival_envelope(grid, start, scen[0], consumables, reserve_s=reserve)

    for r in range(grid.nrows):
        for c in range(grid.ncols):
            re, hv = env.reach_s[r, c], env.haven_s[r, c]
            if not (np.isfinite(re) and np.isfinite(hv)):
                browser_safe = False
            else:
                browser_safe = (env.budget_s - re - hv - reserve) >= 0
            assert browser_safe == bool(env.survivable[r, c]), (r, c)
