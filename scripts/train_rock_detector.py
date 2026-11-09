#!/usr/bin/env python
"""Train the M1 rock detector on the REAL BOULDERING data (headless, GPU).

Standalone version of ``notebooks/train_rock_detector.ipynb`` for running on the
dev GTX 1650 (4 GB): batch 1 + gradient accumulation, AMP, channels_last. Loads
the authors' Detectron2 JSON, trains a torchvision Faster R-CNN on the real
patches, evaluates precision/recall/F1 with the project metric
(`marswalk.ml.metrics_det`), and writes a JSON report + a sample-predictions PNG.

HONESTY: the 265 MB "mask-5px" patches are PNGs without georeferencing, so there
is no ground sample distance (GSD) to turn pixels into metres. We therefore
report **pixel-space** size bins and label them as such — the >=1.5 m gate needs
the georeferenced rasters (the 9.4 GB raw set) or a cited per-product GSD. We
never invent a scale.

    .venv-gpu/Scripts/python.exe scripts/train_rock_detector.py --epochs 12
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Cache pretrained COCO weights on D: (C: is nearly full).
os.environ.setdefault("TORCH_HOME", str(ROOT / ".gpu-build" / "torch-home"))

ROOT_SUBDIR = "Apr2023-Mars-Moon-Earth-mask-5px"
JSON_REL = f"{ROOT_SUBDIR}/json/{ROOT_SUBDIR}.json"
XYXY_ABS, XYWH_ABS = 0, 1


def to_xywh(bbox, mode):
    if mode == XYWH_ABS:
        x, y, w, h = bbox
    else:  # XYXY_ABS
        x1, y1, x2, y2 = bbox
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    return float(x), float(y), float(w), float(h)


def img_path(data_root, split, fn):
    return pathlib.Path(data_root) / ROOT_SUBDIR / "preprocessing" / split / "images" / fn


def load_records(data_root, body_filter=None, limit=0):
    """Read the Detectron2 JSON into light dicts: {id, path, split, boxes[xyxy]}."""
    jp = pathlib.Path(data_root) / JSON_REL
    recs = json.loads(jp.read_text(encoding="utf-8"))
    recs = recs["dataset"] if isinstance(recs, dict) and "dataset" in recs else recs
    prefixes = {"mars": ("ESP", "PSP"), "moon": ("M", "NAC")}
    out = []
    for rec in recs:
        fn = str(rec.get("file_name", ""))
        base = pathlib.Path(fn).name
        if body_filter and not base.upper().startswith(
            tuple(p.upper() for p in prefixes[body_filter])
        ):
            continue
        split = str(rec.get("dataset", "train"))
        p = img_path(data_root, split, base)
        boxes = []
        for a in rec.get("annotations", []):
            x, y, w, h = to_xywh(a["bbox"], a.get("bbox_mode", XYXY_ABS))
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
        out.append({"id": base, "path": str(p), "split": "test" if split != "train" else "train",
                    "boxes": boxes})
        if limit and len(out) >= limit:
            break
    train = [r for r in out if r["split"] == "train"]
    test = [r for r in out if r["split"] == "test"]
    return out, train, test


def read_image(path):
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("L"), dtype="float32")
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "bouldering"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--body", default=None, help="None=all bodies (recommended), or 'mars'")
    ap.add_argument("--limit", type=int, default=0, help="cap #patches (smoke test)")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "runs" / "rock_detector"))
    args = ap.parse_args()

    import torch
    import torchvision
    from torch.utils.data import DataLoader, Dataset

    from marswalk.ml.annotation import BBox
    from marswalk.ml.metrics_det import precision_recall_by_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | device {device} | "
          f"{torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")

    all_recs, train, test = load_records(args.data, args.body, args.limit)
    nbox = sum(len(r["boxes"]) for r in all_recs)
    print(f"patches: {len(all_recs)} (train {len(train)} / test {len(test)}) | boxes: {nbox} "
          f"| body={args.body or 'all'}")
    missing = [r["id"] for r in all_recs if not pathlib.Path(r["path"]).exists()][:3]
    if missing:
        raise SystemExit(f"image files not found, e.g. {missing} — check extraction path "
                         f"({ROOT_SUBDIR}/preprocessing/<split>/images/).")
    if not train or not test:
        raise SystemExit("empty train or test split — check --body / JSON.")

    class DS(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            img = torch.from_numpy(read_image(r["path"])).unsqueeze(0).repeat(3, 1, 1)
            b = r["boxes"]
            if b:
                tgt = {"boxes": torch.tensor(b, dtype=torch.float32),
                       "labels": torch.ones(len(b), dtype=torch.int64)}
            else:
                tgt = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                       "labels": torch.zeros((0,), dtype=torch.int64)}
            return img, tgt, r["id"]

    def collate(x):
        return tuple(zip(*x, strict=True))

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Load the COCO-pretrained detector, then swap the box head for 2 classes
    # (background + rock). Can't pass num_classes=2 with pretrained weights.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model = model.to(device)
    # Patches are ~512 px; keep native resolution (default upscales to 800 and
    # blows the 4 GB budget). Saves VRAM and time.
    model.transform.min_size = (512,)
    model.transform.max_size = 640
    use_amp = device.type == "cuda"
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dl = DataLoader(DS(train), batch_size=args.batch, shuffle=True, collate_fn=collate)

    history, peak_gb, t_start = [], 0.0, time.time()
    model.train()
    for ep in range(args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, run = time.time(), 0.0
        opt.zero_grad()
        for step, (imgs, tgts, _ids) in enumerate(dl):
            imgs = [im.to(device) for im in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = sum(model(imgs, tgts).values()) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            run += float(loss) * args.grad_accum
        ep_loss = run / max(len(dl), 1)
        history.append(ep_loss)
        pk = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        peak_gb = max(peak_gb, pk)
        print(f"epoch {ep+1}/{args.epochs} | loss {ep_loss:.3f} | {time.time()-t0:.1f}s "
              f"| peak VRAM {pk:.2f} GB", flush=True)
    train_s = time.time() - t_start

    # ---- evaluate on the held-out split (pixel-space size bins; GSD unknown) ----
    model.eval()
    agg: dict[str, dict[str, float]] = {}
    samples = []
    with torch.no_grad():
        for r in test:
            img = read_image(r["path"])
            it = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).to(device)
            pred = model([it])[0]
            keep = pred["scores"] >= args.score_thr
            pboxes = [BBox(x=float(b[0]), y=float(b[1]), w=float(b[2]-b[0]), h=float(b[3]-b[1]))
                      for b in pred["boxes"][keep].cpu()]
            pscores = [float(s) for s in pred["scores"][keep].cpu()]
            gboxes = [BBox(x=b[0], y=b[1], w=b[2]-b[0], h=b[3]-b[1]) for b in r["boxes"]]
            res = precision_recall_by_size(pboxes, gboxes, 1.0, iou_thr=args.iou_thr, scores=pscores)
            for k, v in res.items():
                a = agg.setdefault(k, {"tp": 0.0, "fp": 0.0, "fn": 0.0})
                for m in ("tp", "fp", "fn"):
                    a[m] += v[m]
            if len(samples) < 6:
                samples.append((img, r["boxes"],
                                [[b.x, b.y, b.x+b.w, b.y+b.h] for b in pboxes]))

    def prf(a):
        p = a["tp"]/(a["tp"]+a["fp"]) if (a["tp"]+a["fp"]) else 0.0
        rc = a["tp"]/(a["tp"]+a["fn"]) if (a["tp"]+a["fn"]) else 0.0
        return p, rc, (2*p*rc/(p+rc) if (p+rc) else 0.0)

    tot = {"tp": sum(a["tp"] for a in agg.values()),
           "fp": sum(a["fp"] for a in agg.values()),
           "fn": sum(a["fn"] for a in agg.values())}
    op, orc, of1 = prf(tot)
    print(f"\nOVERALL  P {op:.3f}  R {orc:.3f}  F1 {of1:.3f}  (tp/fp/fn "
          f"{tot['tp']:.0f}/{tot['fp']:.0f}/{tot['fn']:.0f})")
    print(f"{'size bin (px)':>16} | {'P':>5} {'R':>5} {'F1':>5}")
    bins_out = {}
    for k, a in agg.items():
        p, rc, f1 = prf(a)
        bins_out[k] = {"precision": p, "recall": rc, "f1": f1,
                       "tp": a["tp"], "fp": a["fp"], "fn": a["fn"]}
        print(f"{k:>16} | {p:.2f} {rc:.2f} {f1:.2f}")

    # ---- write report + sample figure ----
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "BOULDERING (Zenodo 8171052, CC BY 4.0; Prieur et al. 2023)",
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"),
        "body_filter": args.body or "all",
        "n_patches": len(all_recs), "n_train": len(train), "n_test": len(test),
        "n_boxes": nbox,
        "epochs": args.epochs, "batch": args.batch, "grad_accum": args.grad_accum, "lr": args.lr,
        "train_seconds": round(train_s, 1), "peak_vram_gb": round(peak_gb, 2),
        "loss_history": [round(x, 4) for x in history],
        "score_thr": args.score_thr, "iou_thr": args.iou_thr,
        "overall": {"precision": op, "recall": orc, "f1": of1, **tot},
        "size_bins_pixelspace": bins_out,
        "gsd_note": "Patches are non-georeferenced PNGs; size bins are PIXELS, not metres. "
                    "The >=1.5 m gate needs georeferenced rasters or a cited per-product GSD.",
    }
    (out / "rock_detector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out/'rock_detector_report.json'}")

    try:
        from PIL import Image, ImageDraw
        thumbs = []
        for img, gt, pr in samples:
            im = Image.fromarray((img * 255).astype("uint8")).convert("RGB")
            d = ImageDraw.Draw(im)
            for b in gt:
                d.rectangle([b[0], b[1], b[2], b[3]], outline=(80, 220, 100), width=2)
            for b in pr:
                d.rectangle([b[0], b[1], b[2], b[3]], outline=(230, 70, 50), width=2)
            thumbs.append(im)
        if thumbs:
            w = max(t.width for t in thumbs); h = max(t.height for t in thumbs)
            cols = min(3, len(thumbs)); rows = (len(thumbs) + cols - 1) // cols
            canvas = Image.new("RGB", (cols * w + (cols+1)*8, rows * h + (rows+1)*8), (12, 16, 22))
            for i, t in enumerate(thumbs):
                cx = (i % cols) * (w + 8) + 8; cy = (i // cols) * (h + 8) + 8
                canvas.paste(t, (cx, cy))
            canvas.save(out / "rock_detector_samples.png")
            print(f"wrote {out/'rock_detector_samples.png'}  (green=truth, red=prediction)")
    except Exception as e:  # noqa: BLE001
        print(f"(sample figure skipped: {e})")


if __name__ == "__main__":
    main()
