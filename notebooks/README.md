# Marswalk training notebooks — run & return guide

These notebooks train the Marswalk models on **real data** and write back
results. They're meant to be **shared and run by a teammate/friend on a GPU**,
who then sends the outputs to the lead. You don't need to read any code — just
run top to bottom and return two things.

## What you need

- A machine with an **NVIDIA GPU** (4 GB+ is enough for most), or a free cloud GPU:
  - **Google Colab** — Runtime → Change runtime type → **GPU (T4)**, then Run all.
  - **Kaggle** — New Notebook → Settings → Accelerator → **GPU**, then Run all.
  - **Local** — Jupyter with a CUDA PyTorch install.
- Each notebook **downloads its own dataset** (the size is stated in the first
  cell). No manual data setup.

## How to run

1. Open the notebook (`*.ipynb`).
2. On Colab/Kaggle: turn the **GPU on** (above), then **Run all**.
3. Let it finish. The last cells print metrics and draw sample predictions.

## What to send back (the two things)

Every notebook writes, in its output folder:

1. **`<model>_report.json`** — the metrics (e.g. F1 by rock size, per-class IoU,
   loss curve, GPU/VRAM/epoch-time). Download this file.
2. **`<model>_samples.png`** — a few predictions drawn on real images, so we can
   eyeball that it actually learned something.

Send **both** to the lead (Rudresh), plus a one-line note of your GPU name (the
notebook prints it) and whether anything errored. That's it — the lead hands
them to the build and the model goes from "untested" to "verified on real data".

## Who runs what (team laptops)

All seven are now **upload to Colab → Run all** (M1 auto-installs the repo; M2
needs only torch + its dataset; M3–M7 auto-install). Suggested assignment by
hardware:

| Notebook | Model | Assigned laptop | Needs before running |
|---|---|---|---|
| `train_rock_detector.ipynb` | M1 rock detector | **Dev GTX 1650** (running now) — or HP Victus RTX 4050 for a longer run | CUDA GPU ≥4 GB · ~2 GB free disk |
| `train_ai4mars_terrain.ipynb` | M2 surface classifier | **HP Victus — RTX 4050 6 GB** | CUDA GPU (6 GB best) · **~20 GB free disk** · 1–3 h |
| `train_dtm_artifacts.ipynb` | M6 DTM-artifact CNN | **ASUS Zephyrus G14 — RTX 3050 Ti** | GPU (or CPU) · <0.5 GB disk |
| `train_traversability.ipynb` | M3 traversability | **MacBook Air (M4 chip)** | Python 3.10–3.12 + internet (or Colab) |
| `train_dose_regression.ipynb` | M4 dose regression | **MacBook Air (M4 chip)** | Python + internet (or Colab) |
| `train_thermal_emulator.ipynb` | M5 thermal emulator | **MacBook Air (M5 chip)** | Python + internet (or Colab) |
| `train_planner_heuristic.ipynb` | M7 learned heuristic | **MacBook Air (M5 chip)** | Python + internet (or Colab) |

The MacBooks run their four light notebooks on CPU in minutes (no CUDA needed).
Anyone stuck on a local CUDA install should just use **Colab with a GPU runtime**
— it avoids the CPU-vs-CUDA PyTorch problem entirely.

## Honesty / provenance

The data is real and licensed CC BY 4.0 (see [`DATA_LICENSES.md`](../DATA_LICENSES.md)).
The notebooks never fabricate a number — every metric is computed from your run.
If a cell says a value is a **placeholder**, it means exactly that. Results you
return are credited to you and logged in [`AI_LOG.md`](../AI_LOG.md) for the
submission's AI-use disclosure.
