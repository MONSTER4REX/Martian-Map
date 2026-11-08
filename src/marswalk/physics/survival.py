"""Survival envelope: the set of places an explorer can reach AND still walk
back to a safe haven before consumables (minus reserve) run out.

This is the human core of Marswalk and the headline visual: from a start point,
every cell is SURVIVABLE only if

    time_to_reach(start -> cell) + time_to_haven(cell -> nearest haven) + reserve
        <= consumables_time_budget

for the active contingency. Cells that fail are **no-go / death zones** — you
could walk there, but you could not get back in time. Losing the rover (a haven)
or a degraded suit (slower + faster consumption) shrinks the survivable world;
``compare_survival`` quantifies that collapse.

Reach time is a forward Dijkstra from the start; haven time is the multi-source
Dijkstra of ``walkback.time_to_haven_field``. Both use the full-constraint hard
exclusions and the scenario's speed scale. All locomotion/consumables parameters
are PLACEHOLDERS (see those modules); the *mechanism* is what this provides.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from marswalk.physics.consumables import ConsumablesModel
from marswalk.physics.locomotion import LocomotionParams
from marswalk.physics.walkback import (
    WalkbackScenario,
    _allowed_time_edges,
    time_to_haven_field,
)
from marswalk.planner.grid import Cell, Grid
from marswalk.planner.routes import PlannerConfig


def reach_time_field(
    grid: Grid,
    start: Cell,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    speed_scale: float = 1.0,
) -> np.ndarray:
    """Min on-foot time (s) to walk from ``start`` to each cell; inf if blocked."""
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()
    adj: dict[Cell, list[tuple[Cell, float]]] = {}
    for a, b, t in _allowed_time_edges(grid, pcfg, loco, speed_scale):
        adj.setdefault(a, []).append((b, t))

    dist = np.full(grid.shape, math.inf, dtype=float)
    if not grid.is_known(start):
        return dist
    dist[start] = 0.0
    heap: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    counter = 1
    while heap:
        d, _, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for b, t in adj.get(node, ()):  # type: ignore[arg-type]
            nd = d + t
            if nd < dist[b]:
                dist[b] = nd
                heapq.heappush(heap, (nd, counter, b))
                counter += 1
    return dist


@dataclass
class SurvivalEnvelope:
    scenario: str
    budget_s: float
    reserve_s: float
    reach_s: np.ndarray          # time start -> cell
    haven_s: np.ndarray          # time cell -> nearest haven
    margin_s: np.ndarray         # budget - reach - haven - reserve (NaN if unreachable)
    survivable: np.ndarray       # bool: reachable, returnable, margin >= 0
    n_known: int
    n_survivable: int

    @property
    def survivable_fraction(self) -> float:
        return (self.n_survivable / self.n_known) if self.n_known else 0.0

    def known_mask(self) -> np.ndarray:
        return self._base_known

    def no_go(self) -> np.ndarray:
        """Known cells that are NOT survivable (death zones)."""
        return self._base_known & ~self.survivable

    # set by survival_envelope()
    _base_known: np.ndarray = None  # type: ignore[assignment]


def survival_envelope(
    grid: Grid,
    start: Cell,
    scenario: WalkbackScenario,
    consumables: ConsumablesModel,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor: float = 1.0,
) -> SurvivalEnvelope:
    """Compute the survivable world from ``start`` under one contingency."""
    pcfg = pcfg or PlannerConfig()
    loco = loco or LocomotionParams()
    budget = consumables.time_budget_s(activity_factor, scenario.consumption_scale)

    reach = reach_time_field(grid, start, pcfg, loco, scenario.speed_scale)
    haven = time_to_haven_field(grid, scenario.havens, pcfg, loco, scenario.speed_scale)

    with np.errstate(invalid="ignore"):
        margin = budget - reach - haven - reserve_s
    reachable = np.isfinite(reach) & np.isfinite(haven)
    margin = np.where(reachable, margin, np.nan)
    survivable = reachable & (np.nan_to_num(margin, nan=-1.0) >= 0.0)

    base_known = grid.known_mask
    env = SurvivalEnvelope(
        scenario=scenario.name,
        budget_s=budget,
        reserve_s=reserve_s,
        reach_s=reach,
        haven_s=haven,
        margin_s=margin,
        survivable=survivable,
        n_known=int(base_known.sum()),
        n_survivable=int(survivable.sum()),
    )
    env._base_known = base_known
    return env


@dataclass
class SurvivalComparison:
    envelopes: list[SurvivalEnvelope]

    def shrink_vs_first(self) -> list[dict[str, float]]:
        """Per scenario: survivable fraction and the drop vs the first (nominal)."""
        base = self.envelopes[0].n_survivable or 1
        out = []
        for e in self.envelopes:
            lost = self.envelopes[0].n_survivable - e.n_survivable
            out.append({
                "scenario_frac": e.survivable_fraction,
                "cells_survivable": float(e.n_survivable),
                "cells_lost_vs_nominal": float(lost),
                "frac_lost_vs_nominal": lost / base,
            })
        return out


def compare_survival(
    grid: Grid,
    start: Cell,
    scenarios: list[WalkbackScenario],
    consumables: ConsumablesModel,
    pcfg: PlannerConfig | None = None,
    loco: LocomotionParams | None = None,
    reserve_s: float = 1800.0,
    activity_factor: float = 1.0,
) -> SurvivalComparison:
    """Survival envelope under each scenario (nominal first) to show the collapse."""
    envs = [
        survival_envelope(grid, start, s, consumables, pcfg, loco, reserve_s, activity_factor)
        for s in scenarios
    ]
    return SurvivalComparison(envelopes=envs)
