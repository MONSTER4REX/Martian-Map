"""M5: a fast NN surrogate (emulator) of the surface-temperature model.

Thermal screening over a whole AOI, for many times/scenarios, wants a cheap
forward model. This trains a small MLP to reproduce
``physics.thermal.surface_temp_range_k`` — (T_min, T_max) over a diurnal cycle —
from (thermal inertia, noon insolation, albedo). Targets are computed by the
real model, so the emulator is faithful *to that model*.

HONEST SCOPE: the underlying thermal model is a C3 **placeholder** (clear-sky,
flat, unvalidated; see `physics/thermal.py`). The emulator's realism is bounded
by it. The value here is the **mechanism** — swap in a validated thermal model
(KRC / MEDA-calibrated) and retrain, and you get a fast, differentiable surrogate
for routing/time-windows. Trains in seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marswalk.physics.thermal import ThermalParams, surface_temp_range_k

# plausible input ranges (screening scales, not calibrated)
TI_RANGE = (50.0, 800.0)          # thermal inertia (J m^-2 K^-1 s^-1/2)
INSOL_RANGE = (100.0, 700.0)      # noon insolation (W/m^2)
ALBEDO_RANGE = (0.10, 0.40)
FEATURES = ("thermal_inertia", "noon_insolation_wm2", "albedo")


def build_dataset(n: int = 4000, seed: int = 0):
    """Sample inputs and compute (T_min, T_max) with the real thermal model."""
    rng = np.random.default_rng(seed)
    ti = rng.uniform(*TI_RANGE, n)
    ins = rng.uniform(*INSOL_RANGE, n)
    alb = rng.uniform(*ALBEDO_RANGE, n)
    Y = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        tmin, tmax = surface_temp_range_k(ti[i], ins[i], ThermalParams(albedo=float(alb[i])))
        Y[i] = (tmin, tmax)
    X = np.column_stack([ti, ins, alb]).astype(np.float32)
    return X, Y


def _make_net():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2),
    )


@dataclass
class ThermalEmulator:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    state: object = None
    history: list[float] = field(default_factory=list)
    _net: object = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        Xn = (X - self.x_mean) / self.x_std
        with torch.no_grad():
            out = self._net(torch.from_numpy(Xn.astype(np.float32))).cpu().numpy()
        return out * self.y_std + self.y_mean


def train_mlp(X, Y, epochs: int = 200, lr: float = 2e-3, device: str = "cpu",
              batch: int = 512, seed: int = 0) -> ThermalEmulator:
    import torch

    torch.manual_seed(seed)
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-6
    Xn = ((X - x_mean) / x_std).astype(np.float32)
    Yn = ((Y - y_mean) / y_std).astype(np.float32)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    net = _make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    Xt = torch.from_numpy(Xn).to(dev); Yt = torch.from_numpy(Yn).to(dev)
    n = len(X); history = []
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        history.append(tot / n)
    net.eval()
    m = ThermalEmulator(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                        state=net.state_dict(), history=history)
    m._net = net.cpu()
    return m


@dataclass
class EmulatorReport:
    n: int
    mae_tmin_k: float
    mae_tmax_k: float
    max_err_k: float

    def as_dict(self) -> dict:
        return {"n": self.n, "mae_tmin_k": self.mae_tmin_k, "mae_tmax_k": self.mae_tmax_k,
                "max_err_k": self.max_err_k}


def evaluate(model: ThermalEmulator, X, Y) -> EmulatorReport:
    pred = model.predict(X)
    err = np.abs(pred - Y)
    return EmulatorReport(
        n=int(len(Y)),
        mae_tmin_k=float(err[:, 0].mean()),
        mae_tmax_k=float(err[:, 1].mean()),
        max_err_k=float(err.max()),
    )
