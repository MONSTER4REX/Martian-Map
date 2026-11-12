"""M6: detect DEM/DTM artifacts (box-steps, CCD-seam lines) from hillshade.

Real HiRISE DTMs carry known artifacts — box-shaped elevation steps and
CCD-seam lines (`MARSWALK_CONTEXT.md`: "Inspect hillshade; feed this into the
DEM error model"). We have no hand-labelled artifact set, so we use honest
**weak supervision**: inject artifacts of known type into *clean* synthetic DEMs
to make labelled hillshade patches, and train a small CNN to tell clean from
artefacted. An owner then runs it over real DTM hillshades to flag suspect
tiles for the C2 error model. The labels are synthetic by construction (stated
plainly); the detector and its metrics are real.

Self-contained and tiny — trains in seconds on CPU, minutes on any GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marswalk.terrain.synthetic import random_smooth_terrain


def hillshade(elev: np.ndarray, cellsize: float = 10.0,
              az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    """Lambertian hillshade in [0,1] — the view that makes artifacts visible."""
    dzdy, dzdx = np.gradient(elev, cellsize)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    sh = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(sh, 0.0, 1.0)


def inject_box_step(elev: np.ndarray, rng, step_m: float) -> np.ndarray:
    """Add a rectangular constant elevation offset (a DTM box-step artifact)."""
    h, w = elev.shape
    out = elev.copy()
    r0 = rng.integers(0, h // 2); c0 = rng.integers(0, w // 2)
    r1 = rng.integers(r0 + h // 4, h); c1 = rng.integers(c0 + w // 4, w)
    out[r0:r1, c0:c1] += step_m * (1 if rng.random() < 0.5 else -1)
    return out


def inject_seam(elev: np.ndarray, rng, step_m: float) -> np.ndarray:
    """Add a thin linear offset along a row or column (a CCD-seam artifact)."""
    h, w = elev.shape
    out = elev.copy()
    if rng.random() < 0.5:
        c = rng.integers(2, w - 3); out[:, c:c + 2] += step_m   # 2-px seam
    else:
        r = rng.integers(2, h - 3); out[r:r + 2, :] += step_m
    return out


def make_dataset(seeds, patch: int = 32, per_terrain: int = 12, relief_m: float = 25.0,
                 cellsize: float = 10.0, step_frac: float = 0.15):
    """Balanced clean/artefacted hillshade patches.

    For each terrain we cut ``per_terrain`` clean patches (label 0) and the same
    number with a box-step or seam injected (label 1). ``step_frac`` sets the
    artifact height as a fraction of the terrain relief, so it's a realistic
    (not trivially huge) signal. Returns ``X`` (N,1,P,P) float32 and ``y`` (N,).
    """
    rng = np.random.default_rng(0)
    step_m = step_frac * relief_m
    Xs, ys = [], []
    for s in seeds:
        grid = random_smooth_terrain(64, 64, cellsize=cellsize, seed=int(s),
                                     relief_m=relief_m, passes=6)
        elev = grid.elevation
        H, W = elev.shape
        for _ in range(per_terrain):
            r = rng.integers(0, H - patch); c = rng.integers(0, W - patch)
            tile = elev[r:r + patch, c:c + patch]
            Xs.append(hillshade(tile, cellsize)); ys.append(0)
            inj = (inject_box_step if rng.random() < 0.5 else inject_seam)(tile, rng, step_m)
            Xs.append(hillshade(inj, cellsize)); ys.append(1)
    X = np.asarray(Xs, dtype=np.float32)[:, None, :, :]
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def _make_cnn(patch: int):
    import torch.nn as nn

    f = patch // 4
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * f * f, 32), nn.ReLU(), nn.Linear(32, 2),
    )


@dataclass
class ArtifactModel:
    patch: int
    x_mean: float = 0.0
    x_std: float = 1.0
    state: object = None
    history: list[float] = field(default_factory=list)
    _net: object = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xn = (X.astype(np.float32) - self.x_mean) / self.x_std
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xn))
            return logits.argmax(1).cpu().numpy()


def train_cnn(X, y, epochs: int = 15, lr: float = 1e-3, device: str = "cpu",
              batch: int = 64, seed: int = 0) -> ArtifactModel:
    import torch

    torch.manual_seed(seed)
    patch = X.shape[-1]
    x_mean = float(X.mean()); x_std = float(X.std()) + 1e-6
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    net = _make_cnn(patch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xn).to(dev); yt = torch.from_numpy(y).to(dev)
    n = len(y); history = []
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        history.append(tot / n)
    net.eval()
    m = ArtifactModel(patch=patch, x_mean=x_mean, x_std=x_std,
                      state=net.state_dict(), history=history)
    m._net = net.cpu()
    return m


@dataclass
class ArtifactReport:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return {"n": self.n, "accuracy": self.accuracy, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


def evaluate(model: ArtifactModel, X, y) -> ArtifactReport:
    pred = model.predict(X)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = float((pred == y).mean())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return ArtifactReport(n=len(y), accuracy=acc, precision=p, recall=r, f1=f1)
