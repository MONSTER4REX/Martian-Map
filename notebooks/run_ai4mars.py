"""Marswalk M2 — AI4Mars terrain segmentation, headless GPU run.

Faithful to notebooks/train_ai4mars_terrain.ipynb (same loader / model / loss /
IoU), adapted to run from the terminal on the RTX 5050. Config via env vars:

  DATA_DIR    path to extracted ai4mars (default: notebooks/data/ai4mars)
  OUT_DIR     where to write report + weights (default: notebooks)
  IMG_SIZE    default 512
  BATCH       default 4
  EPOCHS      default 1
  MAX_STEPS   cap steps/epoch for a quick fit-check; 0 = full epoch (default 0)
  NUM_WORKERS dataloader workers (default 4)
  LR          default 1e-3
"""
import os, sys, time, json, pathlib
import numpy as np
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

DATA_DIR    = pathlib.Path(os.environ.get("DATA_DIR", "notebooks/data/ai4mars"))
OUT_DIR     = pathlib.Path(os.environ.get("OUT_DIR", "notebooks"))
IMG_SIZE    = int(os.environ.get("IMG_SIZE", 512))
BATCH       = int(os.environ.get("BATCH", 4))
EPOCHS      = int(os.environ.get("EPOCHS", 1))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", 0))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
LR          = float(os.environ.get("LR", 1e-3))

CLASS_NAMES = ["soil", "bedrock", "sand", "big rock"]
NUM_CLASSES = 4
IGNORE = 255
SUBSET = "msl/ncam"
TEST_VERSION = "min1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _subset_base(data_root, subset):
    want = subset.replace("\\", "/")
    for p in pathlib.Path(data_root).rglob("train"):
        s = str(p).replace("\\", "/")
        if s.endswith("/labels/train") and want in s:
            return p.parent.parent
    raise FileNotFoundError(f"could not find {subset}/labels/train under {data_root}")


def _img_for(edr_dir, stem):
    for ext in (".JPG", ".jpg", ".jpeg"):
        p = edr_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_pairs(data_root, subset=SUBSET, test_version=TEST_VERSION):
    base = _subset_base(data_root, subset)
    edr = base / "images" / "edr"
    train = []
    for lbl in (base / "labels" / "train").glob("*.png"):
        img = _img_for(edr, lbl.stem)
        if img:
            train.append((img, lbl))
    tdir = base / "labels" / "test" / f"masked-gold-{test_version}-100agree"
    test = []
    if tdir.exists():
        for lbl in tdir.glob("*.png"):
            img = _img_for(edr, lbl.stem.replace("_merged", ""))
            if img:
                test.append((img, lbl))
    print(f"subset {base.as_posix().split('/')[-2]}/{base.name} | train {len(train)} / "
          f"test {len(test)} (gold {test_version})", flush=True)
    if not train:
        raise RuntimeError("no train pairs — check SUBSET and that the zip extracted.")
    return train, test


def _load_img(path, size):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(a)[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].repeat(3, 1, 1)


def _load_label(path, size):
    a = np.asarray(Image.open(path), dtype=np.int64)
    if a.ndim == 3:
        a = a[..., 0]
    a[(a < 0) | (a >= NUM_CLASSES)] = IGNORE
    t = torch.from_numpy(a)[None, None].float()
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t[0, 0].long()


class AI4MarsDataset(Dataset):
    def __init__(self, pairs, size=IMG_SIZE):
        self.pairs, self.size = pairs, size
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        img_p, lbl_p = self.pairs[i]
        return _load_img(img_p, self.size), _load_label(lbl_p, self.size)


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    m = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=w)
    m.classifier[-1] = torch.nn.Conv2d(256, num_classes, 1)
    m.aux_classifier = None
    return m


def train(model, pairs):
    dl = DataLoader(AI4MarsDataset(pairs), batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, drop_last=len(pairs) >= BATCH,
                    persistent_workers=NUM_WORKERS > 0, pin_memory=True)
    use_amp = device.type == "cuda"
    if use_amp:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    model.train()
    stats = []
    for ep in range(EPOCHS):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, run, nsteps = time.time(), 0.0, 0
        opt.zero_grad()
        for step, (x, y) in enumerate(dl):
            x = x.to(device, memory_format=torch.channels_last) if use_amp else x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)["out"]
                loss = lossfn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            run += float(loss); nsteps += 1
            if step == 0 and device.type == "cuda":
                print(f"  [step 1 ok] VRAM now {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
            if MAX_STEPS and nsteps >= MAX_STEPS:
                break
        peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else float("nan")
        dt = time.time() - t0
        print(f"epoch {ep+1}/{EPOCHS} | loss {run/max(nsteps,1):.3f} | {dt:.1f}s "
              f"| {nsteps} steps | {dt/max(nsteps,1):.3f}s/step | peak VRAM {peak:.2f} GB", flush=True)
        stats.append({"epoch": ep+1, "loss": run/max(nsteps,1), "sec": dt,
                      "steps": nsteps, "sec_per_step": dt/max(nsteps,1), "peak_vram_gb": peak})
    return model, stats


@torch.no_grad()
def evaluate(model, pairs):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    dl = DataLoader(AI4MarsDataset(pairs), batch_size=BATCH, num_workers=NUM_WORKERS)
    for x, y in dl:
        pred = model(x.to(device))["out"].argmax(1).cpu().numpy().ravel()
        tgt = y.numpy().ravel()
        m = tgt != IGNORE
        k = tgt[m] * NUM_CLASSES + pred[m]
        conf += np.bincount(k, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    ious = []
    print(f"{'class':>10} | {'IoU':>6} | pixels", flush=True)
    for c in range(NUM_CLASSES):
        tp = conf[c, c]; fn = conf[c].sum() - tp; fp = conf[:, c].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        ious.append(float(iou))
        print(f"{CLASS_NAMES[c]:>10} | {iou:6.3f} | {conf[c].sum()}", flush=True)
    miou = float(np.nanmean(ious))
    print(f"\nmIoU: {miou:.3f} | big-rock IoU (hazard): {ious[3]:.3f}", flush=True)
    return ious, miou


def main():
    print(f"device {device} | torch {torch.__version__}", flush=True)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print(f"config: IMG_SIZE={IMG_SIZE} BATCH={BATCH} EPOCHS={EPOCHS} "
          f"MAX_STEPS={MAX_STEPS} NUM_WORKERS={NUM_WORKERS} LR={LR}", flush=True)
    train_pairs, test_pairs = load_pairs(DATA_DIR)
    model = build_model().to(device)
    print("model ready:", round(sum(p.numel() for p in model.parameters())/1e6, 1), "M params", flush=True)
    model, stats = train(model, train_pairs)

    ious, miou = (None, None)
    if test_pairs and not MAX_STEPS:
        ious, miou = evaluate(model, test_pairs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "M2 AI4Mars DeepLabV3-ResNet50 terrain segmentation",
        "data": "AI4Mars merged v0.6 (Zenodo 15995036, CC BY 4.0), MSL NavCam",
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "config": {"IMG_SIZE": IMG_SIZE, "BATCH": BATCH, "EPOCHS": EPOCHS,
                   "MAX_STEPS": MAX_STEPS, "NUM_WORKERS": NUM_WORKERS, "LR": LR},
        "train_count": len(train_pairs), "test_count": len(test_pairs),
        "test_version": TEST_VERSION, "epoch_stats": stats,
        "per_class_iou": (dict(zip(CLASS_NAMES, ious)) if ious else None),
        "miou": miou,
        "big_rock_iou": (ious[3] if ious else None),
        "fit_check_only": bool(MAX_STEPS),
    }
    rp = OUT_DIR / "ai4mars_report.json"
    rp.write_text(json.dumps(report, indent=2))
    print("\nwrote report:", rp.resolve(), flush=True)

    if not MAX_STEPS:
        wp = OUT_DIR / "ai4mars_terrain.pt"
        torch.save(model.state_dict(), wp)
        print("wrote weights:", wp.resolve(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
