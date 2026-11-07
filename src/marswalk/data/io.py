"""Windowed COG reads from cloud buckets.

UNTESTED ON REAL DATA: the sandbox has no AWS/PDS access and rasterio may not be
installed. The pure window math (`bounds_to_window`) is tested offline; the
actual read (`read_cog_window`) is guarded behind a rasterio import and must be
exercised by an owner on their machine against real `/vsis3/` or https COGs.

Anonymous AWS Open Data reads: set the rasterio env AWS_NO_SIGN_REQUEST=YES
(handled here). Verify the exact bucket paths in MARSWALK_CONTEXT.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marswalk.planner.grid import Grid

# (a, b, c, d, e, f): x = a*col + b*row + c ; y = d*col + e*row + f  (GDAL order)
AffineTuple = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class WindowSpec:
    col_off: int
    row_off: int
    width: int
    height: int


def bounds_to_window(transform: AffineTuple, bounds: tuple[float, float, float, float]) -> WindowSpec:
    """Pixel window covering a map-coordinate bbox, for a north-up transform.

    `transform` is GDAL-order (a, b, c, d, e, f) with b == d == 0 (north-up).
    `bounds` is (minx, miny, maxx, maxy). The window is clamped to be
    non-negative in size (caller clamps to raster extent when reading).
    """
    a, b, c, d, e, f = transform
    if b != 0 or d != 0:
        raise NotImplementedError("rotated/sheared transforms not supported in bounds_to_window")
    minx, miny, maxx, maxy = bounds
    cols = sorted([(minx - c) / a, (maxx - c) / a])
    rows = sorted([(miny - f) / e, (maxy - f) / e])
    col_off = int(np.floor(cols[0]))
    row_off = int(np.floor(rows[0]))
    width = max(0, int(np.ceil(cols[1])) - col_off)
    height = max(0, int(np.ceil(rows[1])) - row_off)
    return WindowSpec(col_off=col_off, row_off=row_off, width=width, height=height)


def array_to_grid(array: np.ndarray, cellsize: float, nodata: float | None = None) -> Grid:
    """Wrap a raster window as a Grid, mapping nodata to NaN."""
    a = np.asarray(array, dtype=float)
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    return Grid(a, cellsize)


def read_cog_window(
    path: str,
    window: WindowSpec | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    band: int = 1,
    anonymous: bool = True,
):  # pragma: no cover - requires rasterio + network, UNTESTED ON REAL DATA
    """Read a window from a COG. Returns (array, transform_tuple, nodata).

    UNTESTED ON REAL DATA. Requires rasterio. Pass either an explicit `window`
    or map `bounds` (used with the dataset transform). For AWS Open Data, keep
    `anonymous=True` (sets AWS_NO_SIGN_REQUEST).
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:  # keep the message actionable for owners
        raise RuntimeError(
            "read_cog_window requires rasterio (not installed in the sandbox). "
            "Install rasterio and run this on a machine with network access; "
            "this path is UNTESTED ON REAL DATA."
        ) from exc

    env = rasterio.Env(AWS_NO_SIGN_REQUEST="YES") if anonymous else rasterio.Env()
    with env, rasterio.open(path) as ds:
        if window is None:
            if bounds is None:
                raise ValueError("provide either window or bounds")
            transform = (
                ds.transform.a, ds.transform.b, ds.transform.c,
                ds.transform.d, ds.transform.e, ds.transform.f,
            )
            ws = bounds_to_window(transform, bounds)
        else:
            ws = window
        rio_window = Window(ws.col_off, ws.row_off, ws.width, ws.height)
        arr = ds.read(band, window=rio_window)
        win_transform = ds.window_transform(rio_window)
        tt = (win_transform.a, win_transform.b, win_transform.c,
              win_transform.d, win_transform.e, win_transform.f)
        return arr, tt, ds.nodata
