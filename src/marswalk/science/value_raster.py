"""C9 science-value raster: combine informational layers into a value score.

A weighted, min-max-normalized blend of input rasters (e.g. mineralogy
interest, geologic-unit interest, feature proximity). The result is an
INFORMATIONAL/objective raster; `top_targets` turns it into candidate science
targets for the orienteering planner. Inputs are provided by the caller — this
module invents no science values.
"""

from __future__ import annotations

import numpy as np

from marswalk.planner.grid import Cell


def _normalize(a: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] ignoring NaN; constant layers -> 0.5 where finite."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.full(a.shape, np.nan)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    out = np.full(a.shape, np.nan)
    if hi > lo:
        out[finite] = (a[finite] - lo) / (hi - lo)
    else:
        out[finite] = 0.5
    return out


def science_value(layers: dict[str, tuple[np.ndarray, float]]) -> np.ndarray:
    """Weighted blend of normalized layers -> value raster in [0,1].

    `layers` maps name -> (raster, weight). Each raster is min-max normalized,
    then combined by the weights of the layers that are finite at a cell
    (missing layers simply don't contribute). A cell is NaN only where every
    layer is NaN.
    """
    if not layers:
        raise ValueError("need at least one layer")
    shapes = {v[0].shape for v in layers.values()}
    if len(shapes) != 1:
        raise ValueError(f"all layers must share a shape, got {shapes}")
    shape = shapes.pop()

    acc = np.zeros(shape, dtype=float)
    wsum = np.zeros(shape, dtype=float)
    for _name, (raster, weight) in layers.items():
        if weight < 0:
            raise ValueError("weights must be non-negative")
        norm = _normalize(raster)
        finite = np.isfinite(norm)
        acc[finite] += weight * norm[finite]
        wsum[finite] += weight

    out = np.full(shape, np.nan)
    good = wsum > 0
    out[good] = acc[good] / wsum[good]
    return out


def top_targets(
    value: np.ndarray, k: int, min_separation_cells: int = 0
) -> list[tuple[Cell, float]]:
    """Top-k cells by value, optionally enforcing a minimum spacing (greedy)."""
    idx = [
        (int(r), int(c))
        for r in range(value.shape[0])
        for c in range(value.shape[1])
        if np.isfinite(value[r, c])
    ]
    idx.sort(key=lambda rc: value[rc], reverse=True)
    chosen: list[tuple[Cell, float]] = []
    for cell in idx:
        if len(chosen) >= k:
            break
        if min_separation_cells > 0 and any(
            max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) < min_separation_cells
            for p, _ in chosen
        ):
            continue
        chosen.append((cell, float(value[cell])))
    return chosen
