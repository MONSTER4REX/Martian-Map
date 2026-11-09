"""M4: radiation-dose regression (surrogate of the C4 screening model).

Predicts EVA radiation dose (uSv) from (duration, overhead shielding, surface
dose-rate, attenuation length) by learning ``physics.radiation.RadiationScreen
.dose_usv`` across its input space. A fast, differentiable dose field for the
planner's time-windows and route cost.

HONEST SCOPE: the dose model is a C4 **placeholder** (a single GCR-like
background scalar with exponential shielding; no SEP events, spectra, or solar
cycle), and its rate/attenuation are placeholders pending **cited RAD figures**.
The regression is real and faithful to that model; its realism is bounded by it.
Dose spans orders of magnitude, so we learn it in log space. Trains in seconds
on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marswalk.physics.radiation import RadiationScreen

DUR_RANGE = (0.5, 8.0)            # EVA duration (h)
SHIELD_RANGE = (0.0, 60.0)       # overhead shielding (g/cm^2)
RATE_RANGE = (5.0, 50.0)         # surface dose rate (uSv/h) — PLACEHOLDER span
ATTEN_RANGE = (15.0, 45.0)       # attenuation length (g/cm^2) — PLACEHOLDER span
FEATURES = ("duration_h", "shielding_g_cm2", "surface_dose_rate_usv_per_h",
            "attenuation_length_g_cm2")


def dose_of(dur, shield, rate, atten) -> float:
    return RadiationScreen(surface_dose_rate_usv_per_h=rate,
                           attenuation_length_g_cm2=atten).dose_usv(dur, shield)


def build_dataset(n: int = 5000, seed: int = 0):
    """Sample inputs and compute dose (uSv) with the real screening model."""
    rng = np.random.default_rng(seed)
    dur = rng.uniform(*DUR_RANGE, n)
    shield = rng.uniform(*SHIELD_RANGE, n)
    rate = rng.uniform(*RATE_RANGE, n)
    atten = rng.uniform(*ATTEN_RANGE, n)
    y = np.array([dose_of(dur[i], shield[i], rate[i], atten[i]) for i in range(n)],
                 dtype=np.float32)
    X = np.column_stack([dur, shield, rate, atten]).astype(np.float32)
    return X, y


def _make_net():
    import torch.nn as nn

    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 1))


@dataclass
class DoseModel:
    x_mean: np.ndarray
    x_std: np.ndarray
    ylog_mean: float
    ylog_std: float
    state: object = None
    history: list[float] = field(default_factory=list)
    _net: object = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xn = (X - self.x_mean) / self.x_std
        with torch.no_grad():
            out = self._net(torch.from_numpy(Xn.astype(np.float32))).squeeze(-1).cpu().numpy()
        return np.expm1(out * self.ylog_std + self.ylog_mean)   # back to uSv


def train_mlp(X, y, epochs: int = 200, lr: float = 2e-3, device: str = "cpu",
              batch: int = 512, seed: int = 0) -> DoseModel:
    import torch

    torch.manual_seed(seed)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    ylog = np.log1p(y)                                 # dose spans orders of magnitude
    ylog_mean, ylog_std = float(ylog.mean()), float(ylog.std()) + 1e-6
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    Yn = ((ylog - ylog_mean) / ylog_std).astype(np.float32)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    net = _make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    Xt = torch.from_numpy(Xn).to(dev); Yt = torch.from_numpy(Yn).to(dev).unsqueeze(-1)
    n = len(X); history = []
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        history.append(tot / n)
    net.eval()
    m = DoseModel(x_mean=x_mean, x_std=x_std, ylog_mean=ylog_mean, ylog_std=ylog_std,
                  state=net.state_dict(), history=history)
    m._net = net.cpu()
    return m


@dataclass
class DoseReport:
    n: int
    mae_usv: float
    median_rel_err: float

    def as_dict(self) -> dict:
        return {"n": self.n, "mae_usv": self.mae_usv, "median_rel_err": self.median_rel_err}


def evaluate(model: DoseModel, X, y) -> DoseReport:
    pred = model.predict(X)
    rel = np.abs(pred - y) / np.maximum(y, 1e-6)
    return DoseReport(n=int(len(y)), mae_usv=float(np.abs(pred - y).mean()),
                      median_rel_err=float(np.median(rel)))
