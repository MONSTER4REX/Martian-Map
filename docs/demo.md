# Interactive survival console (`web/demo/`)

The judge-facing demo: a Jezero AOI where you **move the cursor over the terrain
and every point is live-scored** — can an astronaut on foot walk there and still
get back to a safe haven before oxygen runs out? Click to pin a destination,
then flip the contingency and watch the safe world collapse.

This is winning-strategy priority #2 (real Mars under the cursor): the thing a
judge *touches*, driven by the same walk-back engine as the hero figure.

## How it stays honest

The browser does **no physics**. All of it is precomputed by the real engine
(`marswalk.physics.survival`) and exported to a static file:

```bash
python scripts/export_jezero_demo.py        # -> web/demo/survival_data.{json,js}
```

For every cell and every contingency we store
`reach_s` (start→cell on-foot time) and `haven_s` (cell→nearest haven). On a
click the page computes exactly the engine's survival rule:

```
margin = budget - reach_s[cell] - haven_s[cell] - reserve
SAFE  <=>  margin >= 0
```

So the SAFE/FATAL verdict and the walk-out / walk-back / reserve / budget
breakdown a judge reads are the engine's numbers, not a re-implementation. The
budget is calibrated from the terrain with the **same recipe** as
`scripts/survival_demo.py`, so the demo and the hero figure agree exactly:
**nominal 80% → rover lost 47% → suit degraded 7%** survivable.

The backdrop is a Lambertian hillshade computed from the DEM. `survival_data.js`
is a tiny `window.SURVIVAL_DATA = …` wrapper so `index.html` opens offline by
double-click (no server, no fetch/CORS).

## View it

```bash
python -m http.server 8731        # then open /web/demo/index.html
```

or just open `web/demo/index.html` directly (the `.js` wrapper loads offline).

## Validity

Terrain is synthetic and all suit/rover/consumables numbers are **PLACEHOLDERS**
(see `physics/locomotion.py` / `physics/consumables.py`). The *mechanism* is
real; an owner's Jezero HiRISE DEM and cited life-support figures drop straight
into `scripts/export_jezero_demo.py` and the map becomes a real site. See
[docs/survival.md](survival.md) for the engine.
