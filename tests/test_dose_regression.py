"""M4 dose-regression tests (offline, CPU, fast).

The surrogate must reproduce the real dose model within a few percent on a
held-out draw (log-space learning handles the wide dynamic range).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from marswalk.ml.dose_regression import build_dataset, dose_of, evaluate, train_mlp  # noqa: E402


def test_dataset_matches_physics():
    X, y = build_dataset(32, seed=1)
    assert X.shape == (32, 4) and y.shape == (32,)
    dur, shield, rate, atten = X[0]
    assert y[0] == pytest.approx(dose_of(dur, shield, rate, atten), rel=1e-5)
    assert np.all(y >= 0)


def test_emulator_accurate_on_holdout():
    Xtr, ytr = build_dataset(5000, seed=0)
    Xte, yte = build_dataset(1000, seed=123)
    model = train_mlp(Xtr, ytr, epochs=200, device="cpu", seed=0)
    assert model.history[-1] < model.history[0]
    rep = evaluate(model, Xte, yte)
    assert rep.median_rel_err < 0.05            # within ~5% of the real model
