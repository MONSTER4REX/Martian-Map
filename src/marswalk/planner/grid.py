"""2.5-D elevation grid. One height per cell; NaN marks unknown/nodata.

Units: elevation in meters, `cellsize` in meters/cell (square cells).
Assumptions A3, A4 in ASSUMPTIONS.md.
"""

from __future__ import annotations

import numpy as np

Cell = tuple[int, int]  # (row, col)


class Grid:
    """A rectangular elevation grid with square cells.

    Parameters
    ----------
    elevation : np.ndarray
        2-D float array, shape (nrows, ncols). ``NaN`` = unknown/nodata.
    cellsize : float
        Ground size of one cell in meters (square cells). Must be > 0.
    """

    __slots__ = ("elevation", "cellsize")

    def __init__(self, elevation: np.ndarray, cellsize: float) -> None:
        arr = np.asarray(elevation, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"elevation must be 2-D, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError("elevation must be non-empty")
        if not (cellsize > 0) or not np.isfinite(cellsize):
            raise ValueError(f"cellsize must be a positive finite number, got {cellsize}")
        self.elevation = arr
        self.cellsize = float(cellsize)

    @property
    def nrows(self) -> int:
        return self.elevation.shape[0]

    @property
    def ncols(self) -> int:
        return self.elevation.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def is_known(self, cell: Cell) -> bool:
        """True if the cell is in bounds and has a finite elevation."""
        if not self.in_bounds(cell):
            return False
        return bool(np.isfinite(self.elevation[cell]))

    def z(self, cell: Cell) -> float:
        return float(self.elevation[cell])

    @property
    def known_mask(self) -> np.ndarray:
        """Boolean array, True where elevation is finite (traversable candidate)."""
        return np.isfinite(self.elevation)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        n_unknown = int((~self.known_mask).sum())
        return (
            f"Grid(shape={self.shape}, cellsize={self.cellsize} m, "
            f"unknown_cells={n_unknown})"
        )
