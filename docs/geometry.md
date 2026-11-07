# C8 geometry: visibility, illumination, comms coverage

Status: **offline baseline complete** on synthetic ephemeris. Owner: P1 (after
Phase 1). Real SPICE path **UNTESTED ON REAL DATA**. Module:
`src/marswalk/geometry/`.

## Frame

Local ENU in meters: a cell is at `(x=col·cs, y=row·cs, z=elevation)`. East =
+x (+col), north = +y (+row). Azimuth is measured from north toward east;
elevation is the angle above local horizontal.

## What it does (`visibility.py`)

- `elevation_azimuth(obs, tgt)` → (elevation°, azimuth°).
- `line_of_sight(grid, obs_cell, target_xyz)` → terrain LOS clearance. Marches
  the ray **only across the AOI footprint** (bounded by the grid diagonal, so a
  far sun/relay is cheap); nodata inside the AOI blocks (unknown ≠ clear).
- `is_visible(…, min_elevation_deg)` → above the elevation mask **and** LOS
  clear.
- `illumination_mask(grid, sun_az, sun_el)` → sunlit boolean mask (sun above
  horizon and not terrain-shadowed); NaN cells unlit.
- `comms_coverage(grid, relay_track, min_elevation_deg, cells=None)` → per-cell
  fraction of relay-track steps with a usable link (score a whole grid or just
  a route's cells).

## Synthetic ephemeris (`ephemeris.py`)

`suggest_relay_pass(...)` — a single orbiter arc rising/setting across the sky;
`sun_direction_point(...)` — a far point in a given sun az/el. Stand-ins for
tests/demos only, **not** real geometry.

## Real SPICE (`spice.py`) — UNTESTED ON REAL DATA

`relay_states(meta_kernel, target, observer, frame, et_times)` wraps
`spiceypy.spkpos`; guarded so it raises an actionable error without spiceypy.
An owner furnishes a NAIF meta-kernel (MRO/MAVEN/Odyssey/TGO — §5), runs it on
their machine, and converts body-fixed positions into the local ENU frame
before feeding the coverage functions.

## Tests

Azimuth/elevation of cardinal targets; LOS clear over flat vs blocked by a
wall; elevation-mask gating; full-lit and no-sun illumination; a wall casts a
shadow on the anti-sun side; coverage fractions stay in [0,1].
