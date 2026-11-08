#!/usr/bin/env python
"""Rock-ablation (§3 gate) demo on a REAL Mars boulder field.

Loads real boulder centres/diameters from the BOULDERING dataset (HiRISE
ESP_017355_2260, Zenodo 8171052, CC BY 4.0 — `examples/bouldering_ESP_017355_2260.json`),
places them on a demo AOI grid, and runs `validation.ablation.rock_ablation`:
slope-only vs rock-aware routing, measuring rock exposure. Prints whether the
gate passes (rock-aware reduces exposure and changes the route) and, with
matplotlib, renders the two routes over the hazard field.

The boulder POSITIONS/SIZES are real; the AOI grid is a demo scale (1 m/cell) —
absolute metres need the HiRISE product GSD, which an owner sets. Run:
    python scripts/rock_ablation_demo.py [-o docs/rock_ablation.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marswalk.planner.grid import Grid
from marswalk.planner.hazard import Rock, rock_exposure, rock_hazard_field
from marswalk.planner.hazard_routes import plan_hazard_route
from marswalk.validation.ablation import rock_ablation

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "bouldering_ESP_017355_2260.json"


def load_rocks(aoi_cells: int, px_per_cell: float, cellsize_m: float):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rocks = []
    for b in data["boulders"]:
        rocks.append(Rock(
            row=b["cy"] / px_per_cell,
            col=b["cx"] / px_per_cell,
            diameter_m=(b["diam_px"] / px_per_cell) * cellsize_m,
        ))
    return rocks, data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs" / "rock_ablation.png")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    AOI, PX_PER_CELL, CELL_M = 100, 5.0, 1.0   # 500 px patch -> 100 cells @ 1 m/cell (demo)
    grid = Grid(np.zeros((AOI, AOI), dtype=float), CELL_M)   # flat: slope never forces a detour
    rocks, meta = load_rocks(AOI, PX_PER_CELL, CELL_M)
    hazard = rock_hazard_field(grid, rocks, min_diameter_m=1.5, buffer_m=1.0)

    start, goal = (50, 2), (50, 97)
    betas = (0.0, 3.0, 8.0, 16.0)
    rep = rock_ablation(grid, hazard, start, goal, betas=betas)

    print(f"REAL boulder field: {meta['patch']} ({meta['n_boulders']} boulders, "
          f"{meta['body']})")
    print(f"AOI {AOI}x{AOI} @ {CELL_M} m/cell (demo scale) | hazard cover "
          f"{(hazard > 0).mean() * 100:.0f}% of cells\n")
    print(f"{'route':>16} | {'len (m)':>8} | {'rock exposure':>13} | changed")
    for e in rep.entries:
        if e.feasible:
            print(f"{e.name:>16} | {e.length_3d_m:8.1f} | {e.rock_exposure:13.1f} | "
                  f"{e.changed_vs_slope_only}")
        else:
            print(f"{e.name:>16} | infeasible")
    print(f"\n{rep.verdict}")

    if args.no_figure:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping figure)"); return

    slope_only = plan_hazard_route(grid, hazard, start, goal, beta=0.0)[0]
    best_beta = max((e.beta for e in rep.entries if e.changed_vs_slope_only), default=betas[-1])
    aware = plan_hazard_route(grid, hazard, start, goal, beta=best_beta)[0]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(hazard, cmap="inferno", origin="upper", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rock hazard [0,1]")
    if slope_only:
        y, x = zip(*slope_only)
        ax.plot(x, y, "--", color="cyan", lw=2.2,
                label=f"slope-only (exposure {rock_exposure(grid, hazard, slope_only):.0f})")
    if aware:
        y, x = zip(*aware)
        ax.plot(x, y, "-", color="lime", lw=2.4,
                label=f"rock-aware β={best_beta:g} (exposure {rock_exposure(grid, hazard, aware):.0f})")
    ax.plot(start[1], start[0], "o", color="white", ms=10, mec="black")
    ax.plot(goal[1], goal[0], "*", color="white", ms=14, mec="black")
    ax.set_title("Rock-aware routing over a REAL Mars boulder field\n"
                 f"BOULDERING {meta['patch']} ({meta['n_boulders']} boulders) — demo AOI")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
