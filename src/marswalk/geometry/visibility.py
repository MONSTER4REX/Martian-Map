"""Line-of-sight visibility, illumination and comms coverage (local ENU frame)."""

from __future__ import annotations

import math

import numpy as np

from marswalk.planner.grid import Cell, Grid

XYZ = tuple[float, float, float]


def cell_xyz(grid: Grid, cell: Cell) -> XYZ:
    r, c = cell
    return (c * grid.cellsize, r * grid.cellsize, float(grid.elevation[cell]))


def elevation_azimuth(observer: XYZ, target: XYZ) -> tuple[float, float]:
    """(elevation_deg, azimuth_deg) of target as seen from observer.

    Azimuth from north (+y) toward east (+x), in [0, 360). Elevation is the
    angle above the local horizontal.
    """
    dx = target[0] - observer[0]
    dy = target[1] - observer[1]
    dz = target[2] - observer[2]
    horiz = math.hypot(dx, dy)
    el = math.degrees(math.atan2(dz, horiz)) if horiz > 0 else (90.0 if dz > 0 else -90.0)
    az = math.degrees(math.atan2(dx, dy)) % 360.0
    return el, az


def _terrain_z_at(grid: Grid, x: float, y: float) -> float:
    """Bilinear terrain elevation at map (x, y); NaN if off-grid or near nodata."""
    cf = x / grid.cellsize
    rf = y / grid.cellsize
    r0 = int(math.floor(rf))
    c0 = int(math.floor(cf))
    if r0 < 0 or c0 < 0 or r0 + 1 >= grid.nrows or c0 + 1 >= grid.ncols:
        if 0 <= r0 < grid.nrows and 0 <= c0 < grid.ncols:
            return float(grid.elevation[r0, c0])
        return math.nan
    tr = rf - r0
    tc = cf - c0
    z = grid.elevation
    z00, z01, z10, z11 = z[r0, c0], z[r0, c0 + 1], z[r0 + 1, c0], z[r0 + 1, c0 + 1]
    if not np.isfinite([z00, z01, z10, z11]).all():
        return math.nan
    top = z00 * (1 - tc) + z01 * tc
    bot = z10 * (1 - tc) + z11 * tc
    return float(top * (1 - tr) + bot * tr)


def _in_footprint(grid: Grid, x: float, y: float) -> bool:
    return 0.0 <= x <= (grid.ncols - 1) * grid.cellsize and 0.0 <= y <= (grid.nrows - 1) * grid.cellsize


def line_of_sight(
    grid: Grid, observer_cell: Cell, target: XYZ, clearance_m: float = 0.0
) -> bool:
    """True if the straight segment observer->target clears the terrain.

    Marches the ray only across the AOI footprint (terrain exists only there);
    beyond the grid nothing blocks. A sample over nodata inside the AOI blocks
    the LOS (unknown is not assumed clear). The step is half a cell, so the
    march is bounded by the grid diagonal regardless of how far the target is.
    """
    ox, oy, oz = cell_xyz(grid, observer_cell)
    tx, ty, tz = target
    total_h = math.hypot(tx - ox, ty - oy)
    if total_h == 0.0:
        return True  # straight up/down: no horizontal terrain to cross
    step = 0.5 * grid.cellsize
    max_h = math.hypot(grid.nrows * grid.cellsize, grid.ncols * grid.cellsize) + step
    th = step
    while th < total_h and th <= max_h:
        t = th / total_h
        x = ox + t * (tx - ox)
        y = oy + t * (ty - oy)
        if not _in_footprint(grid, x, y):
            break  # left the AOI (convex); no terrain beyond blocks the ray
        z_line = oz + t * (tz - oz)
        ground = _terrain_z_at(grid, x, y)
        if math.isnan(ground):
            return False  # nodata inside the AOI blocks (conservative)
        if ground > z_line + clearance_m:
            return False
        th += step
    return True


def is_visible(
    grid: Grid, observer_cell: Cell, target: XYZ, min_elevation_deg: float = 5.0
) -> bool:
    """Target is above the elevation mask AND the terrain LOS is clear."""
    el, _ = elevation_azimuth(cell_xyz(grid, observer_cell), target)
    if el < min_elevation_deg:
        return False
    return line_of_sight(grid, observer_cell, target)


def illumination_mask(
    grid: Grid, sun_azimuth_deg: float, sun_elevation_deg: float, far_m: float = 1e7
) -> np.ndarray:
    """Boolean sunlit mask: sun above the horizon and not terrain-shadowed.

    NaN cells are not illuminated (unknown). The sun is modelled as a far point
    in the given azimuth/elevation direction.
    """
    lit = np.zeros(grid.shape, dtype=bool)
    if sun_elevation_deg <= 0:
        return lit
    az = math.radians(sun_azimuth_deg)
    el = math.radians(sun_elevation_deg)
    dirx = math.sin(az) * math.cos(el)
    diry = math.cos(az) * math.cos(el)
    dirz = math.sin(el)
    for r in range(grid.nrows):
        for c in range(grid.ncols):
            if not math.isfinite(grid.elevation[r, c]):
                continue
            ox, oy, oz = cell_xyz(grid, (r, c))
            sun_pt = (ox + far_m * dirx, oy + far_m * diry, oz + far_m * dirz)
            lit[r, c] = line_of_sight(grid, (r, c), sun_pt)
    return lit


def comms_coverage(
    grid: Grid,
    relay_track: list[XYZ],
    min_elevation_deg: float = 10.0,
    cells: list[Cell] | None = None,
) -> np.ndarray:
    """Per-cell fraction of relay-track steps with a usable link (NaN off-domain).

    If `cells` is given, only those cells are computed (others NaN) — useful to
    score a route cheaply. A step counts if any relay in it is visible.
    """
    out = np.full(grid.shape, np.nan, dtype=float)
    if not relay_track:
        return out
    targets = cells or [
        (r, c)
        for r in range(grid.nrows)
        for c in range(grid.ncols)
        if math.isfinite(grid.elevation[r, c])
    ]
    for cell in targets:
        if not grid.is_known(cell):
            continue
        hits = sum(
            1 for pos in relay_track if is_visible(grid, cell, pos, min_elevation_deg)
        )
        out[cell] = hits / len(relay_track)
    return out
