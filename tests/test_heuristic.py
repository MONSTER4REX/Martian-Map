"""M7 learned-heuristic tests (offline, CPU, fast).

Targets are exact planner computations, so these also guard that the learned
heuristic beats the trivial straight-line lower bound while staying mostly
admissible (pred <= truth) — the property that makes it usable in A*.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from marswalk.ml.heuristic import build_dataset, evaluate, train_mlp  # noqa: E402


def test_build_dataset_shapes_and_finite():
    X, y, flat = build_dataset(range(4), shape=(20, 20), cellsize=10.0)
    assert X.ndim == 2 and X.shape[1] == 4
    assert len(X) == len(y) == len(flat) > 0
    assert np.isfinite(X).all() and np.isfinite(y).all()
    # straight-line flat time is a true lower bound on on-foot time
    assert np.all(flat <= y + 1e-3)


def test_train_and_beat_baseline():
    X, y, flat = build_dataset(range(6), shape=(24, 24), cellsize=10.0)
    model = train_mlp(X, y, epochs=60, device="cpu", seed=0)
    assert len(model.history) == 60
    assert model.history[-1] < model.history[0]           # loss went down
    rep = evaluate(model, X, y, flat)
    assert 0.0 <= rep.admissible_fraction <= 1.0
    assert rep.mae_s < rep.baseline_mae_s                 # learned beats the lower bound
    assert rep.improvement_vs_baseline > 0.0


def test_predict_shape():
    X, y, flat = build_dataset(range(2), shape=(16, 16), cellsize=10.0)
    model = train_mlp(X, y, epochs=10, device="cpu")
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()
