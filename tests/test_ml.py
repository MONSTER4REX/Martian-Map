import importlib.util

import pytest

from marswalk.ml.annotation import AnnotationSet, BBox, LabelSource, RockLabel, Tile
from marswalk.ml.metrics_det import iou, match_detections, precision_recall_by_size
from marswalk.ml.splits import spatial_block_split, splits_are_disjoint
from marswalk.ml.train import TrainConfig, train


def _tile(id_, ox, oy, labels=None):
    return Tile(id=id_, origin_x_m=ox, origin_y_m=oy, px_size_m=0.25, width=256, height=256,
                labels=labels or [])


def test_annotation_roundtrip_and_counts(tmp_path):
    aset = AnnotationSet(tiles=[
        _tile("t1", 0, 0, [
            RockLabel(bbox=BBox(x=10, y=10, w=8, h=6), source=LabelSource.MANUAL),
            RockLabel(bbox=BBox(x=50, y=50, w=4, h=4), source=LabelSource.SAM_PROPOSAL),
        ]),
    ])
    p = tmp_path / "ann.json"
    aset.save(p)
    loaded = AnnotationSet.load(p)
    assert loaded.label_counts() == {"manual": 1, "sam_proposal": 1}
    lab = loaded.by_id("t1").labels[0]
    assert lab.diameter_m(0.25) == pytest.approx(2.0)  # max(8,6)*0.25


def test_spatial_split_is_disjoint():
    # tiles on a grid 64 m apart; blocks of 50 m
    tiles = [_tile(f"t{r}_{c}", c * 64.0, r * 64.0) for r in range(6) for c in range(6)]
    aset = AnnotationSet(tiles=tiles)
    train_ids, test_ids = spatial_block_split(aset, block_size_m=50.0, test_fraction=0.3, seed=1)
    assert train_ids and test_ids
    assert set(train_ids).isdisjoint(test_ids)
    assert splits_are_disjoint(aset, train_ids, test_ids, 50.0)
    # reproducible
    assert spatial_block_split(aset, 50.0, 0.3, 1) == (train_ids, test_ids)


def test_iou_values():
    a = BBox(x=0, y=0, w=10, h=10)
    assert iou(a, a) == pytest.approx(1.0)
    b = BBox(x=10, y=0, w=10, h=10)  # touching, no overlap
    assert iou(a, b) == 0.0
    c = BBox(x=5, y=0, w=10, h=10)  # half overlap -> inter 50, union 150
    assert iou(a, c) == pytest.approx(50 / 150)


def test_matching_tp_fp_fn():
    gts = [BBox(x=0, y=0, w=10, h=10), BBox(x=100, y=100, w=10, h=10)]
    preds = [BBox(x=1, y=1, w=10, h=10), BBox(x=200, y=200, w=10, h=10)]
    matches, fp, fn = match_detections(preds, gts, iou_thr=0.5, scores=[0.9, 0.8])
    assert len(matches) == 1
    assert fp == [1]  # the (200,200) pred matches nothing
    assert fn == [1]  # the (100,100) gt is missed


def test_precision_recall_by_size_bins():
    px = 0.25
    # a big rock (12 px -> 3 m) detected; a small rock (2 px -> 0.5 m) missed
    gts = [BBox(x=0, y=0, w=12, h=12), BBox(x=100, y=100, w=2, h=2)]
    preds = [BBox(x=0, y=0, w=12, h=12)]
    res = precision_recall_by_size(preds, gts, px_size_m=px, iou_thr=0.5, scores=[0.9])
    big = res["[1.5,inf)m"]
    assert big["recall"] == pytest.approx(1.0) and big["precision"] == pytest.approx(1.0)
    small = res["[0.5,1.5)m"]
    assert small["recall"] == 0.0  # the 0.5 m rock (edge of bin) was missed


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch installed; skip the missing-dependency path",
)
def test_train_without_torch_raises():
    with pytest.raises(RuntimeError, match="torch"):
        train(TrainConfig(annotations_json="x.json"))
