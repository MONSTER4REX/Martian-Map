"""Ablation harness: does a feature/parameter actually change the decision?

A feature ships only if an ablation shows a measurable change
(MARSWALK_CONTEXT.md §3). Given named planner configs, plan each to the same
goal and report per-config metrics plus whether the chosen route changed vs a
baseline config (and by how much).
"""

from __future__ import annotations

from pydantic import BaseModel

from marswalk.planner.grid import Cell, Grid
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
