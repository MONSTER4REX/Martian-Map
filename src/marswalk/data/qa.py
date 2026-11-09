"""C1 DTM QA rules: flag no-data, slope spikes, box-steps and CCD seams.

These are FLAGS FOR HUMAN INSPECTION, not corrections. We never fabricate
"fixed" elevations; flagged regions feed the DEM error model (C2) and the
unknown mask. Thresholds are placeholders (document and tune per DTM).

Known Mars-DTM pitfalls handled (MARSWALK_CONTEXT.md §6):
* no-data holes,
* near-vertical slope spikes (processing artifacts),
* box-shaped elevation steps (discontinuities),
* CCD-seam lines (column/row-aligned offsets).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from marswalk.data.derivatives import slope_degrees
from marswalk.planner.grid import Grid


class DTMQAReport(BaseModel):
    shape: tuple[int, int]
    cellsize_m: float
    nodata_count: int
    nodata_fraction: float
    slope_spike_count: int  # cells above slope_spike_deg
    step_edge_count: int  # cells flagged as box-step discontinuities
    seam_columns: list[int]
    seam_rows: list[int]
    thresholds: dict[str, float]
    notes: str


def run_dtm_qa(
    grid: Grid,
    slope_spike_deg: float = 80.0,
    step_m: float = 5.0,
    seam_m: float = 2.0,
) -> tuple[DTMQAReport, dict[str, np.ndarray]]:
    """Return (report, masks). masks has boolean arrays: nodata, slope_spike,
    step_edge. Thresholds are placeholders in DEM units."""
    z = grid.elevation
    nodata = ~np.isfinite(z)

    slope = slope_degrees(grid)
    slope_spike = np.isfinite(slope) & (slope > slope_spike_deg)

    # Box-step discontinuities: large second difference along either axis.
    step_edge = np.zeros_like(z, dtype=bool)
    d2r = np.abs(z[2:, :] - 2 * z[1:-1, :] + z[:-2, :])
    d2c = np.abs(z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2])
    step_edge[1:-1, :] |= np.nan_to_num(d2r, nan=0.0) > step_m
    step_edge[:, 1:-1] |= np.nan_to_num(d2c, nan=0.0) > step_m

    # CCD seams: a column/row boundary where the offset to its neighbour is
    # consistent (large median |difference|) across the whole line.
    seam_cols: list[int] = []
    if grid.ncols >= 2:
        col_diff = np.abs(np.diff(z, axis=1))  # shape (nrows, ncols-1)
        med = np.nanmedian(np.where(np.isfinite(col_diff), col_diff, np.nan), axis=0)
        seam_cols = [int(c + 1) for c in np.where(np.nan_to_num(med, nan=0.0) > seam_m)[0]]

    seam_rows: list[int] = []
    if grid.nrows >= 2:
        row_diff = np.abs(np.diff(z, axis=0))
        med = np.nanmedian(np.where(np.isfinite(row_diff), row_diff, np.nan), axis=1)
        seam_rows = [int(r + 1) for r in np.where(np.nan_to_num(med, nan=0.0) > seam_m)[0]]

    report = DTMQAReport(
        shape=grid.shape,
        cellsize_m=grid.cellsize,
        nodata_count=int(nodata.sum()),
        nodata_fraction=float(nodata.mean()),
        slope_spike_count=int(slope_spike.sum()),
        step_edge_count=int(step_edge.sum()),
        seam_columns=seam_cols,
        seam_rows=seam_rows,
        thresholds={
            "slope_spike_deg": slope_spike_deg,
            "step_m": step_m,
            "seam_m": seam_m,
        },
        notes="Flags for inspection only; no elevations are modified. "
        "Thresholds are placeholders - tune per DTM and feed C2/unknown mask.",
    )
    masks = {"nodata": nodata, "slope_spike": slope_spike, "step_edge": step_edge}
    return report, masks
