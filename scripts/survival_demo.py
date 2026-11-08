#!/usr/bin/env python
"""The hero visual: the astronaut's SURVIVABLE WORLD, and how it collapses.

For a start point and havens (base camp + rover), shows where you can go and
still walk back before consumables (minus reserve) run out — under three
contingencies: nominal, rover lost, suit degraded. Green = survivable (shaded by
spare time); dark red = NO-GO (you could walk there but not get back in time).

Everything is computed by the real walk-back engine (`physics.survival`). The
terrain is synthetic and all suit/rover numbers are PLACEHOLDERS — the owner's
real Jezero DEM and life-support figures drop straight in. Run:
    python scripts/survival_demo.py [-o docs/survival_envelope.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.survival import compare_survival, reach_time_field
from marswalk.physics.walkback import standard_scenarios, time_to_haven_field
from marswalk.terrain.synthetic import random_smooth_terrain

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs" / "survival_envelope.png")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    grid = random_smooth_terrain(60, 60, cellsize=25.0, seed=args.seed, relief_m=60.0, passes=12)
    # The rover is driven OUT near the exploration site; base camp is far. Losing
    # the rover forces the long walk home — that is what collapses the world.
    start, base, rover = (45, 32), (5, 5), (48, 30)
    scen = standard_scenarios(base, rover)              # nominal, rover_loss, suit_degraded
    reserve = 1800.0                                     # 30-min walk-back reserve

    # Calibrate the budget so the NOMINAL world is ~80% survivable (room to watch
    # it collapse) — deterministic from the terrain.
    nom_cost = reach_time_field(grid, start) + time_to_haven_field(grid, scen[0].havens)
    budget = reserve + float(np.percentile(nom_cost[np.isfinite(nom_cost)], 80))
    consumables = ConsumablesModel(items=[Consumable(name="o2", capacity=budget, rate_per_s=1.0)])

    cmp = compare_survival(grid, start, scen, consumables, reserve_s=reserve)

    print(f"budget {budget/60:.0f} min (reserve {reserve/60:.0f} min) | start {start}")
    for e, s in zip(cmp.envelopes, cmp.shrink_vs_first(), strict=True):
        print(f"  {e.scenario:>13}: survivable {e.survivable_fraction*100:5.1f}%  "
              f"({e.n_survivable:4d} cells)  lost vs nominal {s['frac_lost_vs_nominal']*100:5.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("(matplotlib not installed; skipping figure)"); return

    # survivable cells shaded by spare time (min); no-go cells dark red
    cmap = LinearSegmentedColormap.from_list("safe", ["#ffe08a", "#2ecc71", "#0b6b3a"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    titles = {"nominal": "NOMINAL", "rover_loss": "ROVER LOST",
              "suit_degraded": "SUIT DEGRADED"}
    for ax, e in zip(axes, cmp.envelopes, strict=True):
        margin_min = np.where(e.survivable, e.margin_s / 60.0, np.nan)
        base_img = np.zeros((*grid.shape, 3))
        base_img[e.no_go()] = (0.45, 0.05, 0.05)           # death zones: dark red
        base_img[~e.known_mask()] = (0.1, 0.1, 0.1)        # unknown: grey
        ax.imshow(base_img, origin="upper")
        im = ax.imshow(margin_min, cmap=cmap, origin="upper")
        ax.plot(start[1], start[0], "o", color="white", ms=12, mec="black", mew=1.5)
        ax.plot(base[1], base[0], "s", color="cyan", ms=11, mec="black", label="base")
        if "rover_loss" != e.scenario:
            ax.plot(rover[1], rover[0], "^", color="yellow", ms=12, mec="black", label="rover")
        ax.set_title(f"{titles.get(e.scenario, e.scenario)}\nsurvivable "
                     f"{e.survivable_fraction*100:.0f}% of the area", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02, label="spare time to walk back (min)")
    fig.suptitle("Your survivable world on Mars — and how it collapses\n"
                 "green = you can get back · dark red = NO-GO (you can't)   "
                 "[synthetic terrain + PLACEHOLDER life-support; real DEM drops in]",
                 fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
