# P2 rock detector (gated)

Status: schema / splits / metrics **complete and tested offline**; training
**UNTESTED ON REAL DATA** (needs torch + GPU). Owner: P2 (shares the RTX 4050
laptop with P1).

Gates (MARSWALK_CONTEXT.md §3): no ML before the baseline planner passes (it
does); the rock detector ships only if an ablation shows rock-aware routes
reduce rock exposure vs slope-only. Fallback: a literature rock-abundance prior.

## Offline, tested

- `ml/annotation.py` — `AnnotationSet`/`Tile`/`RockLabel`/`BBox` with a
  `LabelSource` (manual / **sam_proposal** / model_pred): SAM labels are
  proposals, not ground truth. Tiles carry map origin + pixel size, so a box's
  real diameter is known. JSON round-trip.
- `ml/splits.py` — `spatial_block_split`: whole spatial blocks go to train or
  test, so splits are **disjoint** (a random tile split leaks via overlap).
  `splits_are_disjoint` verifies it.
- `ml/metrics_det.py` — `iou`, greedy `match_detections`, and
  `precision_recall_by_size` (the gate metric is **F1 for rocks ≥ 1.5 m**, so
  results are per size bin, never a single number).

## Training (UNTESTED)

- `ml/train.py` (`TrainConfig`, `train`) + `scripts/train_rock_detector.py`.
  Targets 6 GB: small pretrained encoder (ResNet18/34, EfficientNet-B0), 256 px
  tiles, batch 2–4, gradient accumulation, `channels_last`, AMP. The model/loop
  is intentionally a stub — an owner implements it and reports back:
  - peak VRAM via `torch.cuda.max_memory_allocated()`,
  - epoch time,
  - precision/recall/**F1 by size bin** on the spatial holdout.
- Pre-cut tiles to disk once; do not run training and horizon ray-march on the
  same GPU at once (§4).

## Training notebook (runnable)

`notebooks/train_rock_detector.ipynb` is a concrete, runnable detector built on
the modules above: torchvision **Faster R-CNN** (class 1 = rock), the spatial
holdout (`ml.splits`), and **F1-by-size** eval (`ml.metrics_det`). It runs
end-to-end on *synthetic* tiles on any machine (CPU included) so the machinery is
verifiable, then you point `cfg.annotations_json` at real labels and set each
`Tile.path`. The loop logs **peak VRAM** and **epoch time**, and the final cell
prints the gate metric (F1 ≥ 1.5 m) plus a "what to paste back" checklist.

```bash
pip install -e ".[nb]"          # torch, torchvision, pillow, jupyter
jupyter notebook notebooks/train_rock_detector.ipynb
```

The data→split→train-step→eval logic was executed on synthetic data (CPU) to
confirm it runs; the **trained result on real data is still UNTESTED**, and ML
stays a capability until the P6 ablation shows rock-aware cost changes routes.
`ml/train.py` remains the owner stub (config + guarded entry point); the notebook
is where the actual loop lives.

Label sources and licenses (Zenodo boulder sets, RMaM-2020 pretraining-only,
AI4Mars license unspecified) are tracked in `DATA_LICENSES.md`.
