"""M5 thermal-emulator tests (offline, CPU, fast).

The emulator must reproduce the real thermal model within a few kelvin on a
held-out sample (different seed) — i.e. it faithfully emulates, not memorizes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from marswalk.ml.thermal_emulator import build_dataset, evaluate, train_mlp  # noqa: E402
from marswalk.physics.thermal import ThermalParams, surface_temp_range_k  # noqa: E402


def test_dataset_matches_physics():
    X, Y = build_dataset(32, seed=1)
    assert X.shape == (32, 3) and Y.shape == (32, 2)
    # a row really is the physics model's output
    ti, ins, alb = X[0]
    tmin, tmax = surface_temp_range_k(ti, ins, ThermalParams(albedo=float(alb)))
    assert Y[0, 0] == pytest.approx(tmin, abs=1e-3)
    assert Y[0, 1] == pytest.approx(tmax, abs=1e-3)
    assert np.all(Y[:, 1] >= Y[:, 0])            # tmax >= tmin always


def test_emulator_accurate_on_holdout():
    Xtr, Ytr = build_dataset(4000, seed=0)
    Xte, Yte = build_dataset(1000, seed=99)      # unseen draw
    model = train_mlp(Xtr, Ytr, epochs=200, device="cpu", seed=0)
    assert model.history[-1] < model.history[0]
    rep = evaluate(model, Xte, Yte)
    assert rep.mae_tmax_k < 4.0                  # within a few K of the real model
    assert rep.mae_tmin_k < 6.0
