#!/usr/bin/env python
"""Export the survival console's fields from REAL Mars boulder data.

Unlike ``export_jezero_demo.py`` (synthetic terrain), this drives the interactive
console (``web/demo/index.html``) with **real orbital data**: the 461 boulder
centres/diameters detected in HiRISE **ESP_017355_2260** (BOULDERING, Prieur
et al. 2023, Zenodo 8171052, CC BY 4.0 — ``examples/bouldering_ESP_017355_2260.json``).

Each real boulder is rendered as a raised rock mound (a boulder IS raised rock:
height ~ its radius -> steep flanks), so the real boulder field becomes a real
micro-relief DEM. The **real walk-back engine** (``physics.survival``) then
computes, for every cell and every contingency,

    reach_s = on-foot time  start -> cell   (slope routes AROUND the boulders)
    haven_s = on-foot time  cell  -> nearest safe haven

The browser does only a lookup + arithmetic on click, so the SAFE / FATAL verdict
is exactly the engine's survival rule over the real boulder field.

Honesty: boulder POSITIONS and SIZES are real; each is modelled as a raised mound
(standard footprint model, not a fabricated DEM). Metres are a demo scale
(absolute metres need the HiRISE product GSD); life-support numbers remain
PLACEHOLDERS. A real regional DEM + cited life-support drop in the same way. Run:

    python scripts/export_boulder_demo.py [-o web/demo/survival_data.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marswalk.physics.consumables import Consumable, ConsumablesModel
from marswalk.physics.survival import reach_time_field, survival_envelope
from marswalk.physics.walkback import standard_scenarios, time_to_haven_field
from marswalk.planner.grid import Grid

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "bouldering_ESP_017355_2260.json"

# Demo AOI: the 500x500 px HiRISE patch -> a 100x100 grid at a demo scale.
AOI = 100
CELL_M = 12.0          # demo metres/cell (absolute metres need the product GSD)
RESERVE_S = 1800.0
MIN_DIAM_M = 3.0       # model boulders >= 3 m as relief (smaller ones are plain)
HEIGHT_FACTOR = 1.0    # mound height = HEIGHT_FACTOR * radius -> ~45 deg+ flanks
LABELS = {"nominal": "NOMINAL", "rover_loss": "ROVER LOST", "suit_degraded": "SUIT DEGRADED"}


def boulder_dem(meta: dict) -> np.ndarray:
    """Real boulders -> a micro-relief DEM. Each boulder is a raised mound whose
    height ~ its radius, so its flanks exceed the planner's max slope and the
    walk-back must route around it. Overlapping boulders take the max (no stacking)."""
    px_per_cell = 500.0 / AOI
    rr, cc = np.mgrid[0:AOI, 0:AOI].astype(float)
    elev = np.zeros((AOI, AOI), dtype=float)
    for b in meta["boulders"]:
        diam_m = (b["diam_px"] / px_per_cell) * CELL_M
        if diam_m < MIN_DIAM_M:
            continue
        row, col = b["cy"] / px_per_cell, b["cx"] / px_per_cell
        rad_cells = max(0.5 * diam_m / CELL_M, 0.6)      # footprint radius (cells)
        height = HEIGHT_FACTOR * 0.5 * diam_m            # mound height (m)
        d2 = (rr - row) ** 2 + (cc - col) ** 2
        mound = height * np.exp(-d2 / (2.0 * (rad_cells * 0.6) ** 2))
        np.maximum(elev, mound, out=elev)
    return elev


def hillshade(elev: np.ndarray, cellsize: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    """Lambertian hillshade in [0, 255] from the DEM (for the backdrop)."""
    dzdy, dzdx = np.gradient(elev, cellsize)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.round(np.clip(shaded, 0.0, 1.0) * 255).astype(int)


def snap_low(elev: np.ndarray, r: int, c: int, thresh: float) -> tuple[int, int]:
    """Nearest cell on open ground (elevation below ``thresh``, i.e. not on a rock)."""
    low = elev < thresh
    if low[r, c]:
        return r, c
    ys, xs = np.where(low)
    k = int(np.argmin((ys - r) ** 2 + (xs - c) ** 2))
    return int(ys[k]), int(xs[k])


def field_to_json(a: np.ndarray) -> list[list[float | None]]:
    return [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "web" / "demo" / "survival_data.json")
    args = ap.parse_args()

    meta = json.loads(FIXTURE.read_text(encoding="utf-8"))
    elev = boulder_dem(meta)
    grid = Grid(elev, CELL_M)

    # Start far from the base, rover parked midway as a second haven -- so losing
    # the rover (rover_loss) forces the long walk to base and the survivable world
    # visibly collapses. Placed on open ground (never on a boulder).
    thresh = MIN_DIAM_M * 0.25
    START = snap_low(elev, 50, 78, thresh)
    BASE = snap_low(elev, 50, 12, thresh)
    ROVER = snap_low(elev, 50, 64, thresh)
    scen = standard_scenarios(BASE, ROVER)

    # Budget calibrated so the NOMINAL world is ~80% survivable (room to collapse),
    # deterministic from the real boulder terrain -- not hand-authored.
    nom_cost = reach_time_field(grid, START) + time_to_haven_field(grid, scen[0].havens)
    finite = nom_cost[np.isfinite(nom_cost)]
    budget = RESERVE_S + float(np.percentile(finite, 80))
    consumables = ConsumablesModel(items=[Consumable(name="o2", capacity=budget, rate_per_s=1.0)])

    scenarios_out = []
    for s in scen:
        env = survival_envelope(grid, START, s, consumables, reserve_s=RESERVE_S)
        scenarios_out.append({
            "name": s.name, "label": LABELS.get(s.name, s.name),
            "budget_s": round(env.budget_s, 1), "reserve_s": RESERVE_S,
            "havens": s.havens, "speed_scale": s.speed_scale,
            "consumption_scale": s.consumption_scale,
            "survivable_fraction": round(env.survivable_fraction, 4),
            "n_survivable": env.n_survivable,
            "reach_s": field_to_json(env.reach_s), "haven_s": field_to_json(env.haven_s),
        })
        print(f"  {s.name:>13}: survivable {env.survivable_fraction*100:5.1f}%  "
              f"budget {env.budget_s/60:.0f} min")

    data = {
        "_note": (f"Precomputed by the real walk-back engine over REAL Mars boulders "
                  f"(BOULDERING {meta['patch']}, {meta['n_boulders']} boulders, CC BY 4.0). "
                  f"Boulders are real raised relief; metres are a demo scale; "
                  f"life-support PLACEHOLDER."),
        "ui": {
            "site": "HiRISE ESP_017355_2260 · ON-FOOT REACHABILITY",
            "provenance": f"Over <b>real HiRISE boulders</b> ({meta['n_boulders']}, "
                          f"ESP_017355_2260) · life-support PLACEHOLDER",
            "honest": (f"Terrain is the <b>real {meta['n_boulders']}-boulder field</b> from "
                       f"HiRISE ESP_017355_2260 (BOULDERING, CC BY 4.0), modelled as raised "
                       f"relief; the <b>mechanism</b> is real, life-support numbers are "
                       f"placeholders. A regional DEM drops in via export_jezero_dem.py."),
        },
        "source": meta.get("source", ""),
        "patch": meta.get("patch", ""), "body": meta.get("body", ""),
        "n_boulders": meta.get("n_boulders"),
        "nrows": AOI, "ncols": AOI, "cellsize": CELL_M,
        "start": list(START), "base": list(BASE), "rover": list(ROVER),
        "reserve_s": RESERVE_S,
        "elev_min": round(float(np.nanmin(elev)), 2),
        "elev_max": round(float(np.nanmax(elev)), 2),
        "hillshade": hillshade(elev, CELL_M).tolist(),
        "known": grid.known_mask.astype(int).tolist(),
        "scenarios": scenarios_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    args.out.write_text(payload, encoding="utf-8")
    js_out = args.out.with_suffix(".js")
    js_out.write_text("window.SURVIVAL_DATA=" + payload + ";\n", encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    n_modelled = int((elev > 0).sum())
    print(f"\nreal boulders: {meta['patch']} ({meta['n_boulders']}) | relief cells "
          f"{n_modelled}/{AOI*AOI} | budget {budget/60:.0f} min")
    print(f"wrote {args.out} ({kb:.0f} KB) + {js_out.name}")


if __name__ == "__main__":
    main()
