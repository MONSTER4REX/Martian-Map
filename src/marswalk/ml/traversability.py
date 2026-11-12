"""M3: traversability from a rover path via Positive-Unlabeled (PU) learning.

A rover's actual drive tells us which ground IS traversable (positive); it never
tells us what is *not* — everywhere it didn't go is **unlabeled** (a mix of
traversable and not). PU learning fits a classifier from positives + unlabeled
and recovers a traversability score for every cell, which the planner can fold
into cost.

Here the "rover path" is a **synthetic traverse** (a random subset of the truly
traversable cells) so the method can be tested offline; wire real Perseverance /
Curiosity waypoints + the co-registered DEM and the same code learns from real
drives. Features are per-cell slope and local roughness (DEM-derived). We report
how well the PU score recovers the held-out *true* traversability (ROC-AUC) — the
true labels are used for evaluation only, never for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marswalk.terrain.synthetic import random_smooth_terrain

FEATURES = ("slope_deg", "roughness_m")


def cell_features(elev: np.ndarray, cellsize: float):
    """Per-cell (slope in degrees, local 3x3 elevation std) — both DEM-derived."""
    dzdy, dzdx = np.gradient(elev, cellsize)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    pad = np.pad(elev, 1, mode="edge")
    acc = np.zeros_like(elev); acc2 = np.zeros_like(elev)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            w = pad[1 + dr:1 + dr + elev.shape[0], 1 + dc:1 + dc + elev.shape[1]]
            acc += w; acc2 += w * w
    mean = acc / 9.0
    rough = np.sqrt(np.maximum(acc2 / 9.0 - mean * mean, 0.0))
    return np.column_stack([slope.ravel(), rough.ravel()]).astype(np.float32)


def make_pu_dataset(seeds, slope_thr_deg: float = 15.0, pos_frac: float = 0.3,
                    cellsize: float = 10.0, relief_m: float = 35.0, seed: int = 0):
    """Features ``X``, PU labels ``s`` (1=rover-visited positive, 0=unlabeled),
    and ``y_true`` (traversable iff slope<=thr) for EVALUATION ONLY."""
    rng = np.random.default_rng(seed)
    Xs, ss, ys = [], [], []
    for sd in seeds:
        grid = random_smooth_terrain(40, 40, cellsize=cellsize, seed=int(sd),
                                     relief_m=relief_m, passes=6)
        X = cell_features(grid.elevation, cellsize)
        slope = X[:, 0]
        y_true = (slope <= slope_thr_deg).astype(np.int64)
        s = np.zeros(len(y_true), dtype=np.int64)
        pos_idx = np.where(y_true == 1)[0]
        if len(pos_idx):
            chosen = rng.choice(pos_idx, size=max(1, int(pos_frac * len(pos_idx))),
                                replace=False)
            s[chosen] = 1
        Xs.append(X); ss.append(s); ys.append(y_true)
    return np.concatenate(Xs), np.concatenate(ss), np.concatenate(ys)


def _make_net():
    import torch.nn as nn

    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                         nn.Linear(16, 1))


@dataclass
class TraversabilityModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    state: object = None
    history: list[float] = field(default_factory=list)
    _net: object = None

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xn = (X - self.x_mean) / self.x_std
        with torch.no_grad():
            logit = self._net(torch.from_numpy(Xn.astype(np.float32))).squeeze(-1)
            return torch.sigmoid(logit).cpu().numpy()


def train_pu(X, s, epochs: int = 60, lr: float = 2e-3, device: str = "cpu",
             batch: int = 1024, seed: int = 0) -> TraversabilityModel:
    """Non-traditional PU: fit positive-vs-unlabeled; the score ranks cells by
    traversability. Unlabeled are treated as negatives (biased but monotone)."""
    import torch

    torch.manual_seed(seed)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    net = _make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xn).to(dev)
    st = torch.from_numpy(s.astype(np.float32)).to(dev).unsqueeze(-1)
    n = len(X); history = []
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), st[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        history.append(tot / n)
    net.eval()
    m = TraversabilityModel(x_mean=x_mean, x_std=x_std, state=net.state_dict(), history=history)
    m._net = net.cpu()
    return m


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum()); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class TraversabilityReport:
    n: int
    roc_auc: float
    n_positive_labeled: int
    true_traversable_frac: float

    def as_dict(self) -> dict:
        return {"n": self.n, "roc_auc": self.roc_auc,
                "n_positive_labeled": self.n_positive_labeled,
                "true_traversable_frac": self.true_traversable_frac}


def evaluate(model: TraversabilityModel, X, s, y_true) -> TraversabilityReport:
    sc = model.score(X)
    return TraversabilityReport(
        n=int(len(y_true)), roc_auc=roc_auc(y_true, sc),
        n_positive_labeled=int(s.sum()),
        true_traversable_frac=float(y_true.mean()),
    )
