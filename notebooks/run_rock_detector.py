"""Headless M1 rock/boulder detector training (REAL BOULDERING data).

Mirrors notebooks/train_rock_detector.ipynb: torchvision Faster R-CNN on the
real Zenodo BOULDERING patches, evaluated with the Marswalk F1-by-size metric
(gate >= 1.5 m). Config via env vars. Writes rock_detector_report.json + .pt.

Run:
  EPOCHS=20 BATCH=4 OUT_DIR=notebooks \
    ./.venv-gpu/Scripts/python.exe notebooks/run_rock_detector.py
"""
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

# ---- config (env) ----
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "data/bouldering"))
OUT_DIR = pathlib.Path(os.environ.get("OUT_DIR", "."))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
LR = float(os.environ.get("LR", "5e-3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))  # >0 = fit-check only
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
BODY_FILTER = os.environ.get("BODY_FILTER") or None
SCORE_THR, IOU_THR = 0.5, 0.5

# make marswalk importable (or installed via pip -e .)
for _c in (pathlib.Path.cwd() / "src", pathlib.Path.cwd().parent / "src"):
    if (_c / "marswalk").exists() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c)); break

from marswalk.ml.annotation import AnnotationSet, BBox, LabelSource, RockLabel, Tile
from marswalk.ml.metrics_det import precision_recall_by_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| device", device, flush=True)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

BOULDERING = {
    "root_subdir": "Apr2023-Mars-Moon-Earth-mask-5px",
    "json_rel": "Apr2023-Mars-Moon-Earth-mask-5px/json/Apr2023-Mars-Moon-Earth-mask-5px.json",
}
BODY_PREFIXES = {"mars": ("ESP", "PSP"), "moon": ("M", "NAC"),
                 "earth": ("courtright", "stonegarden", "southdeadman")}
XYXY_ABS, XYWH_ABS = 0, 1


def _to_xywh(bbox, mode):
    if mode == XYWH_ABS:
        x, y, w, h = bbox
    elif mode == XYXY_ABS:
        x1, y1, x2, y2 = bbox; x, y, w, h = x1, y1, x2 - x1, y2 - y1
    else:
        raise ValueError(f"bbox_mode {mode}: only 0/1 handled")
    return float(x), float(y), float(w), float(h)


def _body_ok(fn, flt):
    if not flt:
        return True
    return fn.upper().startswith(tuple(p.upper() for p in BODY_PREFIXES[flt]))


def _img_path(root, split, fn):
    return pathlib.Path(root) / BOULDERING["root_subdir"] / "preprocessing" / split / "images" / fn


def _read_gsd_m(img_path):
    tif = pathlib.Path(str(img_path).replace("_image.png", "_image.tif"))
    try:
        import rasterio
        with rasterio.open(tif) as ds:
            g = abs(ds.transform.a)
            return g if (ds.crs is not None and g > 0) else None
    except Exception:
        return None


def load_bouldering(data_root, body_filter=BODY_FILTER):
    jp = pathlib.Path(data_root) / BOULDERING["json_rel"]
    recs = json.loads(jp.read_text(encoding="utf-8"))
    recs = recs["dataset"] if isinstance(recs, dict) and "dataset" in recs else recs
    tiles, splits, no_gsd = [], {"train": [], "test": []}, 0
    for rec in recs:
        fn = str(rec.get("file_name", ""))
        if not _body_ok(fn, body_filter):
            continue
        split = str(rec.get("dataset", "train"))
        rid = fn
        img_path = _img_path(data_root, split, fn)
        gsd = _read_gsd_m(img_path)
        if gsd is None:
            no_gsd += 1; gsd = 1.0
        labels = []
        for a in rec.get("annotations", []):
            x, y, w, h = _to_xywh(a["bbox"], a.get("bbox_mode", XYXY_ABS))
            if w > 0 and h > 0:
                labels.append(RockLabel(bbox=BBox(x=x, y=y, w=w, h=h), source=LabelSource.MANUAL))
        tiles.append(Tile(id=rid, path=str(img_path), px_size_m=gsd,
                          width=int(rec.get("width", 0)) or 1,
                          height=int(rec.get("height", 0)) or 1, labels=labels))
        (splits["train"] if split == "train" else splits["test"]).append(rid)
    ann = AnnotationSet(tiles=tiles)
    print(f"tiles: {len(ann.tiles)} | boxes: {sum(len(t.labels) for t in ann.tiles)} | "
          f"train {len(splits['train'])} / test {len(splits['test'])} | no_gsd {no_gsd}", flush=True)
    return ann, splits, no_gsd


def read_image(path):
    try:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype("float32")
    except Exception:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("L"), dtype="float32")
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


class BoulderDataset(Dataset):
    def __init__(self, ann, ids):
        self.tiles = [ann.by_id(i) for i in ids]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        img = read_image(t.path)
        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        boxes = [[l.bbox.x, l.bbox.y, l.bbox.x + l.bbox.w, l.bbox.y + l.bbox.h] for l in t.labels]
        if boxes:
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                      "labels": torch.ones(len(boxes), dtype=torch.int64)}
        else:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros((0,), dtype=torch.int64)}
        return img_t, target, t.id


def collate(b):
    return tuple(zip(*b, strict=True))


def build_model(num_classes=2, pretrained=True):
    # Load COCO-pretrained detector (backbone+FPN+RPN), then replace the box head
    # for our 2 classes (background + boulder). torchvision forbids passing
    # num_classes alongside pretrained weights, so swap the predictor instead.
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    w = "DEFAULT" if pretrained else None
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=w)
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return m


def train(model, ann, ids):
    dl = DataLoader(BoulderDataset(ann, ids), batch_size=BATCH, shuffle=True,
                    collate_fn=collate, num_workers=NUM_WORKERS)
    use_amp = device.type == "cuda"
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    stats = []
    for ep in range(EPOCHS):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, run, nsteps = time.time(), 0.0, 0
        opt.zero_grad()
        for step, (imgs, targets, _ids) in enumerate(dl):
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = sum(model(imgs, targets).values()) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            run += float(loss) * GRAD_ACCUM; nsteps += 1
            if ep == 0 and step == 0:
                vram = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                print(f"  [step 1 ok] VRAM now {vram:.2f} GB", flush=True)
            if MAX_STEPS and nsteps >= MAX_STEPS:
                break
        peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else float("nan")
        dt = time.time() - t0
        print(f"epoch {ep+1}/{EPOCHS} | loss {run/max(nsteps,1):.3f} | {dt:.1f}s | "
              f"{nsteps} steps | peak VRAM {peak:.2f} GB", flush=True)
        stats.append({"epoch": ep + 1, "loss": run / max(nsteps, 1), "sec": dt,
                      "steps": nsteps, "peak_vram_gb": peak})
        if MAX_STEPS:
            break
    return model, stats


@torch.no_grad()
def evaluate(model, ann, ids):
    model.eval()
    agg = {}
    for tid in ids:
        t = ann.by_id(tid)
        img = read_image(t.path)
        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).to(device)
        pred = model([img_t])[0]
        keep = pred["scores"] >= SCORE_THR
        pboxes = [BBox(x=float(b[0]), y=float(b[1]), w=float(b[2] - b[0]), h=float(b[3] - b[1]))
                  for b in pred["boxes"][keep].cpu()]
        pscores = [float(s) for s in pred["scores"][keep].cpu()]
        res = precision_recall_by_size([*pboxes], [l.bbox for l in t.labels], t.px_size_m,
                                       iou_thr=IOU_THR, scores=pscores)
        for k, v in res.items():
            a = agg.setdefault(k, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
            for m in ("tp", "fp", "fn"):
                a[m] += v[m]
    rows = {}
    print(f"{'size bin':>14} | {'P':>5} {'R':>5} {'F1':>5} | tp/fp/fn", flush=True)
    for k, a in agg.items():
        p = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
        r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows[k] = {"precision": p, "recall": r, "f1": f1,
                   "tp": a["tp"], "fp": a["fp"], "fn": a["fn"]}
        print(f"{k:>14} | {p:.2f} {r:.2f} {f1:.2f} | {a['tp']:.0f}/{a['fp']:.0f}/{a['fn']:.0f}", flush=True)
    gate = rows.get("[1.5,inf)m", {}).get("f1", 0.0)
    print(f"\nGATE - F1 for boulders >= 1.5 m: {gate:.3f}  (aspirational >= 0.8)", flush=True)
    return rows, gate


def main():
    print(f"launching M1 run | EPOCHS={EPOCHS} BATCH={BATCH} GRAD_ACCUM={GRAD_ACCUM} "
          f"LR={LR} MAX_STEPS={MAX_STEPS} WORKERS={NUM_WORKERS} body={BODY_FILTER or 'all'}", flush=True)
    ann, splits, no_gsd = load_bouldering(DATA_DIR)
    train_ids, test_ids = splits["train"], splits["test"]
    if not train_ids or not test_ids:
        raise RuntimeError("empty split — check data / BODY_FILTER")
    model = build_model().to(device)
    print("model ready:", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params", flush=True)
    model, stats = train(model, ann, train_ids)
    rows, gate = evaluate(model, ann, test_ids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "M1 BOULDERING Faster R-CNN ResNet50-FPN (torchvision)",
        "data": "BOULDERING (Prieur et al. 2023, Zenodo 8171052, CC BY 4.0)",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "config": {"EPOCHS": EPOCHS, "BATCH": BATCH, "GRAD_ACCUM": GRAD_ACCUM, "LR": LR,
                   "MAX_STEPS": MAX_STEPS, "BODY_FILTER": BODY_FILTER or "all"},
        "train_count": len(train_ids), "test_count": len(test_ids),
        "tiles_without_gsd": no_gsd,
        "epoch_stats": stats,
        "f1_by_size": rows,
        "gate_f1_1p5m": gate,
        "fit_check_only": bool(MAX_STEPS),
    }
    rp = OUT_DIR / "rock_detector_report.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote report:", rp.resolve(), flush=True)
    if not MAX_STEPS:
        wp = OUT_DIR / "rock_detector.pt"
        torch.save(model.state_dict(), wp)
        print("wrote weights:", wp.resolve(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
