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

## Training notebook (real data)

`notebooks/train_rock_detector.ipynb` trains a boulder detector on the **real
BOULDERING dataset** (Prieur et al., 2023 — [doi:10.5281/zenodo.8171052](https://doi.org/10.5281/zenodo.8171052),
**CC BY 4.0**). No synthetic data. It:

- downloads the real 265 MB masks/JSON bundle (MD5-verified) — optionally the
  655 MB pretrained model and 9.4 GB rasters;
- loads the dataset's **Detectron2 custom-dataset JSON** into our schema (953
  patches, ~39 k boulder boxes; Mars/HiRISE `ESP_` subset = 49 patches, ~6.8 k
  boxes), honouring the authors' train/validation/test split (`dataset` field);
- trains torchvision **Faster R-CNN** (AMP + `channels_last` + grad accumulation),
  logging **peak VRAM** and **epoch time**;
- evaluates with **F1-by-size** (`ml.metrics_det`, gate ≥ 1.5 m).

Body is in the filename prefix (`ESP_`/`PSP_` = Mars, `M*`/`NAC_` = Moon, place
names = Earth); the set is built for cross-body transfer, so training on all
bodies and evaluating on Mars is the intended use. Diameter→metre bins need a
real **GSD**, read from each `.tif` (rasterio / the `geo` extra); the loader
**warns and never invents a scale** if it can't read one.

```bash
pip install -e ".[nb]"          # torch, torchvision, pillow, jupyter
# optional, for real GSD:  pip install -e ".[geo]"   # rasterio
jupyter notebook notebooks/train_rock_detector.ipynb
```

Verified here against the real download: JSON load, split, and a train/eval step
all run on the actual patches (CPU). The **trained result is still owner-side**
(needs the GPU + full run), and ML stays a capability until the P6 ablation shows
rock-aware cost changes routes. **Cite Prieur et al. (2023), CC BY 4.0.**
`ml/train.py` remains the owner config-stub; the notebook holds the real loop.

## AI4Mars terrain segmentation (real data)

`notebooks/train_ai4mars_terrain.ipynb` is a *separate* task: per-pixel terrain
**semantic segmentation** on the real **AI4Mars** dataset (Swan, Atha, Ono et
al. — [doi:10.5281/zenodo.15995036](https://doi.org/10.5281/zenodo.15995036),
**CC BY 4.0**, ~16 GB). Rover-view (navcam), not orbital, so it classifies what
the rover sees rather than giving a top-down hazard raster. It:

- downloads the real 16 GB merged dataset (MD5-verified);
- loads **MSL NavCam** (`msl/ncam`) image/label pairs with the NAV class key
  **verified from the dataset's `label_keys.json`** — `0` soil, `1` bedrock,
  `2` sand, `3` big rock, `255` NULL/ignore — using the authors' `train/` crowd
  split and `test/masked-gold-min{1,2,3}-100agree/` expert gold;
- trains torchvision **DeepLabV3-ResNet50** (CrossEntropy `ignore_index=255`,
  AMP + `channels_last`), logging peak VRAM + epoch time;
- reports **per-class IoU + mIoU**, highlighting **big-rock IoU** (the
  routing-relevant hazard class).

```bash
pip install -e ".[nb]"
jupyter notebook notebooks/train_ai4mars_terrain.ipynb
```

Verified here on **real extracted MSL NavCam patches** (8 train / 4 test,
range-read from the 16 GB archive without downloading it all): load → train →
per-class IoU all run on CPU. Full training is owner-side (16 GB + GPU).
`M2020_GEO` uses a different geology key (values by tens) and is skipped.
**Cite Swan/Atha/Ono et al. (CVPRW 2021 / IEEE Aero 2022).**

Label sources and licenses — BOULDERING (CC BY 4.0), AI4Mars (CC BY 4.0 on
Zenodo; NASA-portal note unresolved), RMaM-2020 (pretraining-only) — are tracked
in `DATA_LICENSES.md`.
