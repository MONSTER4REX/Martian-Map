#!/usr/bin/env python
"""Export the survival engine's fields to a static JSON the web demo reads.

The interactive demo (``web/demo/index.html``) lets a judge click any point on
the map and instantly see "SAFE, +Xm margin" or "FATAL -- can't get back". The
heavy lifting is done HERE by the real walk-back engine (``physics.survival``):
for every cell and every contingency we precompute

    reach_s  = on-foot time  start -> cell
    haven_s  = on-foot time  cell  -> nearest safe haven

The browser then only does a lookup + arithmetic on click:

    margin = budget - reach_s[cell] - haven_s[cell] - reserve
    SAFE  <=> margin >= 0

so the verdict a judge sees is exactly the engine's survival rule, not a
re-implementation. Nothing is hand-authored: budget is calibrated from the
terrain (same recipe as ``scripts/survival_demo.py`` -> 80/47/7).

Terrain is synthetic and all suit/rover/consumables numbers are PLACEHOLDERS
(see ``physics/locomotion.py`` / ``physics/consumables.py``); an owner's real
Jezero DEM and cited life-support figures drop straight in. Run:

    python scripts/export_jezero_demo.py [-o web/demo/survival_data.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.survival import reach_time_field, survival_envelope
from marswalk.physics.walkback import standard_scenarios, time_to_haven_field
from marswalk.terrain.synthetic import random_smooth_terrain

ROOT = Path(__file__).resolve().parent.parent

# Keep in lock-step with scripts/survival_demo.py so the demo and the hero
# figure tell the same numbers.
GRID = dict(nrows=60, ncols=60, cellsize=25.0, seed=7, relief_m=60.0, passes=12)
START, BASE, ROVER = (45, 32), (5, 5), (48, 30)
RESERVE_S = 1800.0
LABELS = {"nominal": "NOMINAL", "rover_loss": "ROVER LOST", "suit_degraded": "SUIT DEGRADED"}


def hillshade(elev: np.ndarray, cellsize: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    """Standard Lambertian hillshade in [0, 255] from the DEM (for the backdrop)."""
    dzdy, dzdx = np.gradient(elev, cellsize)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shaded = np.clip(shaded, 0.0, 1.0)
    return np.round(shaded * 255).astype(int)


def field_to_json(a: np.ndarray) -> list[list[float | None]]:
    """Row-major array of rounded seconds; inf/NaN -> null (unreachable)."""
    out: list[list[float | None]] = []
    for row in a:
        out.append([None if not np.isfinite(v) else round(float(v), 1) for v in row])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "web" / "demo" / "survival_data.json")
    args = ap.parse_args()

    grid = random_smooth_terrain(
        GRID["nrows"], GRID["ncols"], cellsize=GRID["cellsize"],
        seed=GRID["seed"], relief_m=GRID["relief_m"], passes=GRID["passes"],
    )
    scen = standard_scenarios(BASE, ROVER)

    # Budget calibrated so the NOMINAL world is ~80% survivable (room to collapse)
    # -- deterministic from the terrain, identical to the hero figure.
    nom_cost = reach_time_field(grid, START) + time_to_haven_field(grid, scen[0].havens)
    budget = RESERVE_S + float(np.percentile(nom_cost[np.isfinite(nom_cost)], 80))
    consumables = ConsumablesModel(items=[Consumable(name="o2", capacity=budget, rate_per_s=1.0)])

    scenarios_out = []
    for s in scen:
        env = survival_envelope(grid, START, s, consumables, reserve_s=RESERVE_S)
        scenarios_out.append({
            "name": s.name,
            "label": LABELS.get(s.name, s.name),
            "budget_s": round(env.budget_s, 1),
            "reserve_s": RESERVE_S,
            "havens": s.havens,
            "speed_scale": s.speed_scale,
            "consumption_scale": s.consumption_scale,
            "survivable_fraction": round(env.survivable_fraction, 4),
            "n_survivable": env.n_survivable,
            "reach_s": field_to_json(env.reach_s),
            "haven_s": field_to_json(env.haven_s),
        })
        print(f"  {s.name:>13}: survivable {env.survivable_fraction*100:5.1f}%  "
              f"budget {env.budget_s/60:.0f} min")

    elev = grid.elevation
    data = {
        "_note": ("Precomputed by the real walk-back engine (marswalk.physics.survival). "
                  "Synthetic terrain + PLACEHOLDER life-support; real Jezero DEM drops in."),
        "nrows": grid.nrows, "ncols": grid.ncols, "cellsize": grid.cellsize,
        "start": list(START), "base": list(BASE), "rover": list(ROVER),
        "reserve_s": RESERVE_S,
        "elev_min": round(float(np.nanmin(elev)), 2),
        "elev_max": round(float(np.nanmax(elev)), 2),
        "hillshade": hillshade(elev, grid.cellsize).tolist(),
        "known": grid.known_mask.astype(int).tolist(),
        "scenarios": scenarios_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    args.out.write_text(payload, encoding="utf-8")
    # Also emit a JS wrapper so index.html opens offline (file://) without a
    # fetch -- a judge can double-click the page with no server.
    js_out = args.out.with_suffix(".js")
    js_out.write_text("window.SURVIVAL_DATA=" + payload + ";\n", encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} ({kb:.0f} KB) + {js_out.name} | budget {budget/60:.0f} min "
          f"| grid {grid.nrows}x{grid.ncols} @ {grid.cellsize} m")


if __name__ == "__main__":
    main()
