"""Ablation harness: does a feature/parameter actually change the decision?

A feature ships only if an ablation shows a measurable change
(MARSWALK_CONTEXT.md §3). Given named planner configs, plan each to the same
goal and report per-config metrics plus whether the chosen route changed vs a
baseline config (and by how much).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from marswalk.planner.grid import Cell, Grid
from marswalk.planner.hazard import rock_exposure
from marswalk.planner.hazard_routes import plan_hazard_route
from marswalk.planner.routes import PlannerConfig, plan_route
from marswalk.validation.metrics import hazard_exposure, jaccard, path_length_3d


class AblationEntry(BaseModel):
    name: str
    feasible: bool
    length_3d_m: float | None
    hazard_exposure: int | None
    jaccard_to_baseline: float | None
    changed_vs_baseline: bool


class AblationReport(BaseModel):
    baseline: str
    policy: str
    entries: list[AblationEntry]

    def changed(self) -> list[str]:
        return [e.name for e in self.entries if e.changed_vs_baseline]


def ablation(
    grid: Grid,
    start: Cell,
    goal: Cell,
    configs: dict[str, PlannerConfig],
    baseline: str,
    policy: str = "full_constraint",
    change_jaccard_threshold: float = 0.999,
) -> AblationReport:
    """Plan each config; flag those whose route differs from the baseline's."""
    if baseline not in configs:
        raise ValueError(f"baseline {baseline!r} not in configs {list(configs)}")

    paths: dict[str, list[Cell] | None] = {}
    for name, cfg in configs.items():
        paths[name] = plan_route(grid, start, goal, policy, cfg)[0]

    base_path = paths[baseline]
    entries: list[AblationEntry] = []
    for name, cfg in configs.items():
        p = paths[name]
        if p is None:
            entries.append(
                AblationEntry(
                    name=name,
                    feasible=False,
                    length_3d_m=None,
                    hazard_exposure=None,
                    jaccard_to_baseline=(0.0 if base_path is not None else None),
                    changed_vs_baseline=(base_path is not None),
                )
            )
            continue
        j = jaccard(base_path, p) if base_path is not None else None
        changed = (base_path is None) or (j is not None and j < change_jaccard_threshold)
        entries.append(
            AblationEntry(
                name=name,
                feasible=True,
                length_3d_m=path_length_3d(grid, p),
                hazard_exposure=hazard_exposure(grid, p, cfg.max_slope_deg),
                jaccard_to_baseline=j,
                changed_vs_baseline=bool(changed) and name != baseline,
            )
        )
    return AblationReport(baseline=baseline, policy=policy, entries=entries)


# --------------------------------------------------------------------------- #
# Rock-aware ablation — the MARSWALK_CONTEXT.md §3 gate for shipping the
# rock detector: rock-aware routing must REDUCE rock exposure (and change the
# route) vs slope-only, else the rock layer does not earn its place.
# --------------------------------------------------------------------------- #
class RockAblationEntry(BaseModel):
    name: str
    beta: float
    feasible: bool
    length_3d_m: float | None
    rock_exposure: float | None        # integrated hazard along the route (hazard·m)
    jaccard_to_slope_only: float | None
    changed_vs_slope_only: bool


class RockAblationReport(BaseModel):
    entries: list[RockAblationEntry]
    gate_passed: bool                  # rock-aware reduced exposure AND changed route
    exposure_reduction_frac: float | None  # (slope_only - best) / slope_only
    verdict: str

    def by_beta(self, beta: float) -> RockAblationEntry:
        for e in self.entries:
            if e.beta == beta:
                return e
        raise KeyError(beta)


def rock_ablation(
    grid: Grid,
    hazard: np.ndarray,
    start: Cell,
    goal: Cell,
    config: PlannerConfig | None = None,
    betas: tuple[float, ...] = (0.0, 2.0, 5.0),
    max_hazard: float | None = None,
    change_jaccard_threshold: float = 0.999,
) -> RockAblationReport:
    """Plan at each ``beta`` (0 = slope-only baseline) and test the §3 gate.

    Rock exposure is measured on the SAME hazard field for every route, so the
    comparison is like-for-like. The gate passes if some rock-aware route both
    changes the route and lowers rock exposure vs the slope-only baseline.
    """
    cfg = config or PlannerConfig()
    if 0.0 not in betas:
        betas = (0.0, *betas)

    paths: dict[float, list[Cell] | None] = {}
    for b in betas:
        paths[b] = plan_hazard_route(grid, hazard, start, goal, cfg, b, max_hazard)[0]

    base_path = paths[0.0]
    base_exposure = rock_exposure(grid, hazard, base_path) if base_path else None

    entries: list[RockAblationEntry] = []
    best_exposure = base_exposure
    for b in betas:
        p = paths[b]
        if p is None:
            entries.append(RockAblationEntry(
                name=("slope_only" if b == 0.0 else f"rock_aware_b{b:g}"),
                beta=b, feasible=False, length_3d_m=None, rock_exposure=None,
                jaccard_to_slope_only=(0.0 if base_path else None),
                changed_vs_slope_only=bool(base_path)))
            continue
        exp = rock_exposure(grid, hazard, p)
        j = jaccard(base_path, p) if base_path else None
        changed = (base_path is None) or (j is not None and j < change_jaccard_threshold)
        if b != 0.0 and best_exposure is not None and exp < best_exposure:
            best_exposure = exp
        entries.append(RockAblationEntry(
            name=("slope_only" if b == 0.0 else f"rock_aware_b{b:g}"),
            beta=b, feasible=True, length_3d_m=path_length_3d(grid, p),
            rock_exposure=exp, jaccard_to_slope_only=j,
            changed_vs_slope_only=bool(changed) and b != 0.0))

    reduced = (
        base_exposure is not None and best_exposure is not None
        and best_exposure < base_exposure
    )
    any_changed = any(e.changed_vs_slope_only for e in entries)
    gate = bool(reduced and any_changed)
    frac = ((base_exposure - best_exposure) / base_exposure
            if (reduced and base_exposure) else (0.0 if base_exposure else None))
    if base_path is None:
        verdict = "slope-only route infeasible; cannot assess the rock gate."
    elif gate:
        verdict = (f"GATE PASSED: rock-aware routing cut rock exposure by "
                   f"{frac * 100:.0f}% and changed the route.")
    elif reduced:
        verdict = "exposure reduced but route unchanged within threshold."
    else:
        verdict = ("GATE NOT MET: rock-aware routing did not reduce rock exposure "
                   "here (no boulder field on the slope-only route).")
    return RockAblationReport(entries=entries, gate_passed=gate,
                              exposure_reduction_frac=frac, verdict=verdict)
