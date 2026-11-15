#!/usr/bin/env python
"""Export the survival console's fields from a REAL Mars DEM (MOLA, Jezero).

This drives the interactive console (``web/demo/index.html``) with **real
elevation**: a 100x100 window of MGS MOLA MEGDR topography (128 ppd, 463 m/px)
centred on Jezero crater / the Perseverance landing site, extracted from PDS tile
``megt44n000hb`` and committed as ``examples/jezero_mola_dem_463m.json``.

The **real walk-back engine** (``physics.survival``) runs on this real DEM: for
every cell and every contingency it precomputes, from the real slopes,

    reach_s = on-foot time  start -> cell   (routes around terrain too steep to cross)
    haven_s = on-foot time  cell  -> nearest safe haven

The browser does only a lookup + arithmetic on click, so the SAFE / FATAL verdict
is exactly the engine's survival rule over real Jezero topography.

Honesty: the elevation is REAL MOLA data (463 m/px, so this is a ~46 km regional
traverse, not a short walking EVA); the O2 budget is calibrated from the terrain,
and life-support rate/reserve numbers remain PLACEHOLDERS until cited figures drop
in. Run:

    python scripts/export_jezero_dem.py [-o web/demo/survival_data.json]
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
FIXTURE = ROOT / "examples" / "jezero_mola_dem_463m.json"
RESERVE_S = 1800.0
LABELS = {"nominal": "NOMINAL", "rover_loss": "ROVER LOST", "suit_degraded": "SUIT DEGRADED"}


def hillshade(elev, cellsize, az_deg=315.0, alt_deg=45.0):
    dzdy, dzdx = np.gradient(elev, cellsize)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.round(np.clip(shaded, 0.0, 1.0) * 255).astype(int)


def snap_reachable(reachable, r, c):
    if reachable[r, c]:
        return r, c
    ys, xs = np.where(reachable)
    k = int(np.argmin((ys - r) ** 2 + (xs - c) ** 2))
    return int(ys[k]), int(xs[k])


def field_to_json(a):
    return [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "web" / "demo" / "survival_data.json")
    ap.add_argument("--js-var", default="SURVIVAL_DATA", help="global the .js wrapper assigns to")
    args = ap.parse_args()

    dem = json.loads(FIXTURE.read_text(encoding="utf-8"))
    elev = np.asarray(dem["elevation"], dtype=float)
    cellsize = float(dem["cellsize_m"])
    nrows, ncols = elev.shape
    grid = Grid(elev, cellsize)

    # Reachable region from a seed on the crater floor (lowest ground), so start,
    # base and rover are always mutually connected (not split by a steep wall).
    seed = tuple(int(x) for x in np.unravel_index(int(np.argmin(elev)), elev.shape))
    reachable = np.isfinite(reach_time_field(grid, seed))

    # Base (haven) and start spread across the floor, rover parked between them
    # as a second haven, so losing it (rover_loss) collapses the survivable world.
    START = snap_reachable(reachable, nrows // 2, int(ncols * 0.82))
    BASE = snap_reachable(reachable, nrows // 2, int(ncols * 0.18))
    ROVER = snap_reachable(reachable, nrows // 2, int(ncols * 0.62))
    scen = standard_scenarios(BASE, ROVER)

    # Budget calibrated so NOMINAL is ~80% survivable -- deterministic from the
    # real terrain, not hand-authored.
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
        "_note": (f"Precomputed by the real walk-back engine over a REAL Mars DEM: "
                  f"{dem['source']}, {dem['center']}. Real MOLA elevation at "
                  f"{cellsize:.0f} m/px (regional scale); life-support PLACEHOLDER."),
        "ui": {
            "short": "MOLA DEM",
            "site": "MOLA · JEZERO CRATER · REGIONAL REACHABILITY",
            "provenance": "Real <b>MOLA topography</b> (MGS MEGDR, 463 m/px) over "
                          "<b>Jezero crater</b> · life-support PLACEHOLDER",
            "honest": "Terrain is <b>real MOLA elevation</b> of Jezero crater "
                      "(MGS MOLA MEGDR, 463 m/px) — a ~46 km regional traverse; the "
                      "<b>mechanism</b> is real, life-support numbers are placeholders. "
                      "Cited O₂ figures drop straight in.",
        },
        "source": dem["source"], "dem_center": dem["center"],
        "lat_north": dem.get("lat_north"), "lat_south": dem.get("lat_south"),
        "lon_west": dem.get("lon_west"), "lon_east": dem.get("lon_east"),
        "nrows": nrows, "ncols": ncols, "cellsize": cellsize,
        "start": list(START), "base": list(BASE), "rover": list(ROVER),
        "reserve_s": RESERVE_S,
        "elev_min": round(float(elev.min()), 1), "elev_max": round(float(elev.max()), 1),
        "hillshade": hillshade(elev, cellsize).tolist(),
        "known": grid.known_mask.astype(int).tolist(),
        "scenarios": scenarios_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    args.out.write_text(payload, encoding="utf-8")
    js_out = args.out.with_suffix(".js")
    js_out.write_text(f"window.{args.js_var}=" + payload + ";\n", encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    print(f"\nreal DEM: {dem['center']} | elev {elev.min():.0f}..{elev.max():.0f} m "
          f"| {nrows}x{ncols} @ {cellsize:.0f} m/px | budget {budget/60:.0f} min")
    print(f"wrote {args.out} ({kb:.0f} KB) + {js_out.name}")


if __name__ == "__main__":
    main()
