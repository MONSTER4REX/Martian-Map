# Model roadmap — restored PRD v0.1 scope (6 trained + 1 optional)

The lead (Rudresh) restored the **six-model** scope from PRD v0.1. PRD v0.2 had
trimmed this to "1 core + 1 gated"; we are bringing the five cut models back as
**trainable, shareable notebooks**. This file is the honest status of each.

> **Workflow (unchanged, `MARSWALK_CONTEXT.md` §6):** Claude writes the notebook
> and verifies it *runs* (offline self-check, and on real data where the sandbox
> can reach it); an owner or a friend runs the full training on a real GPU and
> sends the outputs back. A model is **not "done"** until someone pastes back real
> metrics. Every notebook therefore ends by writing a `*_report.json` + sample
> figures that are easy to screenshot and return — see
> [`notebooks/README.md`](../notebooks/README.md).

## Hardware reality

| Machine | GPU | Use |
|---|---|---|
| This dev laptop | **GTX 1650, 4 GB** | M1 trainable here (small batch); others verify-run |
| Team (plan) | RTX 4050 6 GB / RTX 3050 Ti 4 GB | detector training, ablations |
| Friends | their own GPUs / Colab / Kaggle | run notebooks, return outputs |

4 GB tips (from the plan): ResNet18/34 or EfficientNet-B0 encoder, 256 px tiles,
batch 2–4, gradient accumulation, `channels_last`, AMP. Pre-cut tiles once.

## The six (+1) models

| # | Model | Real data | Status | Trains on 4 GB? |
|---|---|---|---|---|
| **M1** | **Rock detector** (Faster R-CNN) | BOULDERING — Zenodo 8171052, CC BY 4.0 | notebook ✅; **training on the 1650 now** | yes (small) |
| **M2** | **Surface classifier** (DeepLabV3) | AI4Mars — Zenodo 15995036, CC BY 4.0 | notebook ✅ (verified-run); needs 16 GB + GPU | tight |
| **M3** | **Traversability** (PU-learning) | rover traverse mask + DEM/terrain features | ✅ module + tests + notebook (CPU-verified) | yes |
| **M4** | **Dose regression** | MSL/RAD reference + shielding geometry | ✅ module + tests + notebook (CPU-verified); data thin — inputs marked | yes |
| **M5** | **Thermal emulator** (surrogate NN) | trained on the physics thermal model's own outputs | ✅ module + tests + notebook (CPU-verified) | yes |
| **M6** | **DTM-artifact CNN** | HiRISE hillshade; labels by injected box-step/CCD-seam artifacts | ✅ module + tests + notebook (CPU-verified) | yes |
| **M7** | **Learned planner heuristic** (optional) | self-generated: A* cost-to-go on many terrains | ✅ module + tests + notebook (CPU-verified) | yes (CPU ok) |

**Honesty flags.** M1/M2 train on real public labels. M5/M6/M7 are
self-contained and *real in mechanism* — M5 emulates our own physics model, M6
learns from artifacts we inject into clean DEMs (a standard weak-supervision
trick), M7 learns from our own planner's solutions. M3 is best with real rover
traverse data (Perseverance/Curiosity waypoints); until that is wired it runs on
a documented synthetic traverse, clearly marked. M4 has the weakest public
dataset; its pipeline is real but its inputs are **placeholders** pending cited
RAD figures. Nothing prints a number we did not compute.

## Gates (kept from the plan)

A model ships into the product only if it passes its gate (`MARSWALK_CONTEXT.md`
§96): rock-aware cost needs an ablation win (M1 — already passing on real
boulders); the surface classifier needs a route-decision improvement (M2);
thermal needs matched-quantity MEDA validation (M5); radiation needs a stated
validity range (M4). A model that fails its gate is cut or hidden — restoring
the scope does **not** lower the bar for shipping.
