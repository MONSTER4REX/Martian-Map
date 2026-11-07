"""Detection metrics: IoU matching and precision/recall/F1 by rock size bin.

The gate metric is F1 for rocks >= 1.5 m (MARSWALK_CONTEXT.md §3), so results
are reported per size bin, not as one number. Pure numpy; works on predicted
and ground-truth boxes without any ML framework.
"""

from __future__ import annotations

from marswalk.ml.annotation import BBox


def iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    preds: list[BBox],
    gts: list[BBox],
    iou_thr: float = 0.5,
    scores: list[float] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match preds->gts by descending score. Returns (matches, fp_pred_idx,
    fn_gt_idx) where matches is a list of (pred_idx, gt_idx)."""
    order = sorted(
        range(len(preds)),
        key=lambda i: (scores[i] if scores is not None else 0.0),
        reverse=True,
    )
    used_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    fp: list[int] = []
    for pi in order:
        best_j, best_iou = -1, iou_thr
        for gj in range(len(gts)):
            if gj in used_gt:
                continue
            v = iou(preds[pi], gts[gj])
            if v >= best_iou:
                best_iou, best_j = v, gj
        if best_j >= 0:
            used_gt.add(best_j)
            matches.append((pi, best_j))
        else:
            fp.append(pi)
    fn = [gj for gj in range(len(gts)) if gj not in used_gt]
    return matches, fp, fn


def _bin_of(diameter_m: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= diameter_m < edges[i + 1]:
            return i
    return len(edges) - 2 if diameter_m >= edges[-1] else -1


def precision_recall_by_size(
    preds: list[BBox],
    gts: list[BBox],
    px_size_m: float,
    size_edges_m: list[float] | None = None,
    iou_thr: float = 0.5,
    scores: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 per diameter bin. TP/FN binned by GT size, FP by pred
    size. `size_edges_m` default edges: [0, 0.5, 1.5, inf)."""
    edges = size_edges_m or [0.0, 0.5, 1.5, float("inf")]
    nb = len(edges) - 1
    tp = [0] * nb
    fp = [0] * nb
    fn = [0] * nb

    def diam(b: BBox) -> float:
        return max(b.w, b.h) * px_size_m

    matches, fp_idx, fn_idx = match_detections(preds, gts, iou_thr, scores)
    for _pi, gj in matches:
        b = _bin_of(diam(gts[gj]), edges)
        if b >= 0:
            tp[b] += 1
    for pi in fp_idx:
        b = _bin_of(diam(preds[pi]), edges)
        if b >= 0:
            fp[b] += 1
    for gj in fn_idx:
        b = _bin_of(diam(gts[gj]), edges)
        if b >= 0:
            fn[b] += 1

    out: dict[str, dict[str, float]] = {}
    for i in range(nb):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        hi = edges[i + 1]
        label = f"[{edges[i]:g},{'inf' if hi == float('inf') else f'{hi:g}'})m"
        out[label] = {
            "precision": p, "recall": r, "f1": f1,
            "tp": float(tp[i]), "fp": float(fp[i]), "fn": float(fn[i]),
        }
    return out
