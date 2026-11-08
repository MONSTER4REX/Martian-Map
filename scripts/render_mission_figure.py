#!/usr/bin/env python
"""Render the mission demo as a 3-panel figure (offline, from the real pipeline).

Panels:
  A  terrain hillshade + the three routes (shortest / slope-aware / mission),
     with start, goal and walk-back havens marked.
  B  walk-back time-to-haven field (rover-loss contingency) + the mission route.
  C  route-stability frequency map over a DEM error ensemble + baseline route.

Nothing here is hand-authored; every path/field comes from the marswalk modules.
Requires matplotlib (`pip install -e ".[viz]"`). Offline, synthetic terrain;
UNTESTED ON REAL DATA (all physics/threshold params are PLACEHOLDERS).

Run:
    python scripts/render_mission_figure.py [-o docs/mission_figure.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from marswalk.data.derivatives import hillshade
from marswalk.data.error_model import DEMErrorModel
from marswalk.physics.consumables import BASELINE_CONSUMABLES
from marswalk.physics.locomotion import LocomotionParams
from marswalk.physics.walkback import standard_scenarios, time_to_haven_field
from marswalk.planner.mission import plan_walkback_route
from marswalk.planner.routes import PlannerConfig, compare_routes
from marswalk.terrain.synthetic import ridge_world
from marswalk.validation.stability import route_stability

ROOT = Path(__file__).resolve().parent.parent


def _xy(path):
    """Route cells (row, col) -> plot (x=col, y=row) arrays."""
    rows = [c[0] for c in path]
    cols = [c[1] for c in path]
    return cols, rows


def render(out_path: Path, stability_n: int = 40, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcfg = PlannerConfig()
    loco = LocomotionParams()
    grid = ridge_world(40, 40, cellsize=2.0, ridge_height_m=2.0,
                       ridge_half_width=1, gap_col=20)
    start, goal = (0, 0), (39, 0)
    base, rover = start, goal

    cmp = compare_routes(grid, start, goal, pcfg)
    scenarios = {s.name: s for s in standard_scenarios(base, rover)}
    rover_loss = scenarios["rover_loss"]

    wb = plan_walkback_route(grid, start, goal, rover_loss, BASELINE_CONSUMABLES, pcfg, loco)
    tth = time_to_haven_field(grid, rover_loss.havens, pcfg, loco, rover_loss.speed_scale)
    tth_min = tth / 60.0
    tth_min[~np.isfinite(tth_min)] = np.nan

    model = DEMErrorModel(sill_m2=0.25, range_m=10.0)
    stab = route_stability(grid, start, goal, model, n=stability_n, pcfg=pcfg, seed=seed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # --- Panel A: hillshade + three routes ---
    ax = axes[0]
    ax.imshow(hillshade(grid), cmap="gray", origin="upper")
    styles = {
        "Shortest (3-D distance)": ("red", "shortest (rejected)"),
        "Slope-aware": ("orange", "slope-aware (rejected)"),
        "Full-constraint (mission)": ("lime", "mission route"),
    }
    for name, (color, label) in styles.items():
        r = cmp.by_name(name)
        if r.path:
            x, y = _xy(r.path)
            ls = "-" if r.mission_feasible else "--"
            ax.plot(x, y, ls, color=color, lw=2.4, label=label)
    for cell, mark, txt, col in ((start, "o", "start", "white"),
                                 (goal, "*", "goal", "white"),
                                 (base, "s", "base", "cyan"),
                                 (rover, "^", "rover", "yellow")):
        ax.plot(cell[1], cell[0], mark, color=col, ms=11, mec="black", mew=1.0)
    ax.set_title("A. Terrain + three routes")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_xlabel("col"); ax.set_ylabel("row")

    # --- Panel B: walk-back time-to-haven field + mission route ---
    ax = axes[1]
    im = ax.imshow(tth_min, cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="walk-back time (min)")
    if wb.feasible and wb.path:
        x, y = _xy(wb.path)
        ax.plot(x, y, "-", color="white", lw=2.2, label="walk-back-safe route")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.plot(base[1], base[0], "s", color="cyan", ms=11, mec="black")
    ax.set_title("B. Walk-back time to haven (rover-loss)")
    ax.set_xlabel("col"); ax.set_ylabel("row")

    # --- Panel C: route-stability frequency map ---
    ax = axes[2]
    im = ax.imshow(stab.frequency, cmap="magma", origin="upper", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of replans through cell")
    if stab.baseline_path:
        x, y = _xy(stab.baseline_path)
        ax.plot(x, y, "-", color="cyan", lw=1.8, label="baseline route")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"C. Route stability ({stab.n_feasible}/{stab.n} runs, "
        f"mean Jaccard {stab.mean_jaccard:.2f})"
    )
    ax.set_xlabel("col"); ax.set_ylabel("row")

    fig.suptitle(
        "Marswalk Planner - end-to-end demo (synthetic terrain; PLACEHOLDER params; "
        "UNTESTED ON REAL DATA)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs" / "mission_figure.png")
    ap.add_argument("--stability-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = render(args.out, args.stability_n, args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
