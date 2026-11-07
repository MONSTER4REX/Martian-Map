"""Neighbor offsets for 8- and 16-connected lattices (Assumption A5)."""

from __future__ import annotations

# 8-connectivity: the king moves.
_OFFSETS_8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# 16-connectivity: king moves + knight-style moves to reduce metrication error.
_OFFSETS_16: tuple[tuple[int, int], ...] = _OFFSETS_8 + (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2),  (1, 2),  (2, -1),  (2, 1),
)


def neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    """Return (dr, dc) offsets for the given connectivity (8 or 16)."""
    if connectivity == 8:
        return _OFFSETS_8
    if connectivity == 16:
        return _OFFSETS_16
    raise ValueError(f"connectivity must be 8 or 16, got {connectivity}")
