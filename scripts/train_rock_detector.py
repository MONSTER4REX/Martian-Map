#!/usr/bin/env python
"""CLI wrapper for the rock-detector training run. UNTESTED ON REAL DATA.

Run on the RTX 4050 laptop (needs torch + CUDA):

    python scripts/train_rock_detector.py --annotations data/rocks/ann.json

See docs/ml.md for the recipe and what to paste back (peak VRAM, epoch time,
F1 by size bin on the spatial holdout).
"""

from __future__ import annotations

import argparse

from marswalk.ml.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Marswalk rock detector")
    ap.add_argument("--annotations", required=True, help="annotations JSON (ml.annotation)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--out-dir", default="runs/rock_detector")
    args = ap.parse_args()

    cfg = TrainConfig(
        annotations_json=args.annotations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        encoder=args.encoder,
        out_dir=args.out_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
