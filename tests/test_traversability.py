"""M3 PU traversability tests (offline, CPU, fast).

Trained only on positives + unlabeled (no true labels), the score must still
recover the held-out *true* traversability well (ROC-AUC) on unseen terrain.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from marswalk.ml.traversability import (  # noqa: E402
    evaluate,
    make_pu_dataset,
    roc_auc,
    train_pu,
)


def test_auc_helper_matches_known_cases():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(np.array([0, 1]), np.array([0.9, 0.1])) == pytest.approx(0.0)


def test_pu_dataset_shapes_and_labels():
    X, s, y = make_pu_dataset(range(3))
    assert X.shape[1] == 2 and len(X) == len(s) == len(y)
    assert set(np.unique(s)).issubset({0, 1}) and s.sum() > 0
    # every labeled positive is genuinely traversable (positives are a subset)
    assert np.all(y[s == 1] == 1)


def test_pu_recovers_traversability_on_unseen_terrain():
    Xtr, str_, ytr = make_pu_dataset(range(8))
    Xte, ste, yte = make_pu_dataset(range(100, 104))
    model = train_pu(Xtr, str_, epochs=60, device="cpu", seed=0)
    assert model.history[-1] < model.history[0]
    rep = evaluate(model, Xte, ste, yte)
    assert rep.roc_auc > 0.85            # recovers true traversability from positives only
