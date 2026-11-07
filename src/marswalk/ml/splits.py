"""Spatially-disjoint train/test splits (no leakage between nearby tiles).

Tiles are assigned to square spatial blocks; whole blocks go to train or test,
so a test tile never neighbours a train tile within the same block. This is the
holdout the gates require — a random tile split would leak because adjacent
tiles overlap in context.
"""

from __future__ import annotations

import numpy as np

from marswalk.ml.annotation import AnnotationSet, Tile


def _block_of(tile: Tile, block_size_m: float) -> tuple[int, int]:
    cx, cy = tile.center_xy_m()
    return (int(np.floor(cx / block_size_m)), int(np.floor(cy / block_size_m)))


def spatial_block_split(
    annotations: AnnotationSet,
    block_size_m: float,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) by holding out whole spatial blocks."""
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")
    blocks: dict[tuple[int, int], list[str]] = {}
    for t in annotations.tiles:
        blocks.setdefault(_block_of(t, block_size_m), []).append(t.id)

    keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test_blocks = max(1, round(len(keys) * test_fraction))
    test_blocks = set(keys[:n_test_blocks])

    train_ids: list[str] = []
    test_ids: list[str] = []
    for k in keys:
        (test_ids if k in test_blocks else train_ids).extend(blocks[k])
    return sorted(train_ids), sorted(test_ids)


def splits_are_disjoint(
    annotations: AnnotationSet,
    train_ids: list[str],
    test_ids: list[str],
    block_size_m: float,
) -> bool:
    """True if no spatial block contains both a train and a test tile."""
    by_id = {t.id: t for t in annotations.tiles}
    train_blocks = {_block_of(by_id[i], block_size_m) for i in train_ids}
    test_blocks = {_block_of(by_id[i], block_size_m) for i in test_ids}
    return train_blocks.isdisjoint(test_blocks)
