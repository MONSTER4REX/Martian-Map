"""M6 DTM-artifact detector tests (offline, CPU, fast).

Train and test terrains are disjoint, so a pass means the CNN generalizes to
unseen terrain rather than memorizing — the property that matters for flagging
artifacts in real DTMs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from marswalk.ml.dtm_artifacts import evaluate, make_dataset, train_cnn  # noqa: E402


def test_dataset_balanced_and_shaped():
    X, y = make_dataset(range(4), patch=32, per_terrain=10)
    assert X.shape[1:] == (1, 32, 32)
    assert len(X) == len(y) == 4 * 10 * 2
    assert y.mean() == pytest.approx(0.5)            # balanced clean/artefacted
    assert np.isfinite(X).all() and X.min() >= 0.0 and X.max() <= 1.0


def test_detects_injected_artifacts_on_unseen_terrain():
    Xtr, ytr = make_dataset(range(20), patch=32, per_terrain=16, step_frac=0.3)
    Xte, yte = make_dataset(range(100, 106), patch=32, per_terrain=16, step_frac=0.3)  # disjoint
    model = train_cnn(Xtr, ytr, epochs=25, device="cpu", seed=0)
    assert model.history[-1] < model.history[0]
    rep = evaluate(model, Xte, yte)
    assert rep.accuracy > 0.75                        # well above chance (0.5)
    assert rep.f1 > 0.7
