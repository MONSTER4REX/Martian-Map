"""Route stability under DEM perturbation (the headline uncertainty figure).

Replan the route across a DEM error ensemble (C2) and report how often the
recommendation changes. A stable recommendation is one that barely moves under
plausible DEM error; an unstable one is driven by error and must be shown as
such — never presented as a single certain route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from marswalk.data.error_model import DEMErrorModel, ensemble
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig, plan_route
from marswalk.validation.metrics import jaccard

Planner = Callable[[Grid], list[Cell] | None]


@dataclass
class StabilityReport:
    baseline_path: list[Cell] | None
    n: int
    n_feasible: int
    n_infeasible: int
    mean_jaccard: float  # mean cell-set similarity of ensemble routes to baseline
    frac_identical: float  # fraction of ensemble routes identical to baseline
    frequency: np.ndarray  # per-cell fraction of ensemble routes passing through

    def summary(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "n_feasible": float(self.n_feasible),
            "n_infeasible": float(self.n_infeasible),
            "mean_jaccard": self.mean_jaccard,
            "frac_identical": self.frac_identical,
        }


def route_stability(
    grid: Grid,
    start: Cell,
    goal: Cell,
    model: DEMErrorModel,
    n: int = 50,
    plan: Planner | None = None,
    pcfg: PlannerConfig | None = None,
    seed: int = 0,
) -> StabilityReport:
    """Perturb the DEM `n` times, replan, and measure route agreement."""
    pcfg = pcfg or PlannerConfig()
    if plan is None:
        def plan(g: Grid) -> list[Cell] | None:
            return plan_route(g, start, goal, "full_constraint", pcfg)[0]

    baseline = plan(grid)
    freq = np.zeros(grid.shape, dtype=float)
    jac: list[float] = []
    identical = 0
    n_feasible = 0

    for g in ensemble(grid, model, n, seed):
        p = plan(g)
        if p is None:
            continue
        n_feasible += 1
        for cell in set(p):
            freq[cell] += 1
        if baseline is not None:
            j = jaccard(baseline, p)
            jac.append(j)
            if p == baseline:
                identical += 1

    if n_feasible:
        freq /= n_feasible
    return StabilityReport(
        baseline_path=baseline,
        n=n,
        n_feasible=n_feasible,
        n_infeasible=n - n_feasible,
        mean_jaccard=float(np.mean(jac)) if jac else 0.0,
        frac_identical=(identical / n_feasible) if n_feasible else 0.0,
        frequency=freq,
    )
