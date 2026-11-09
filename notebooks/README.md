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

## The notebooks

| Notebook | Model | Dataset (auto-download) | Rough time on a T4 |
|---|---|---|---|
| `train_rock_detector.ipynb` | M1 rock detector | BOULDERING, ~265 MB | 15–40 min |
| `train_ai4mars_terrain.ipynb` | M2 surface classifier | AI4Mars, ~16 GB | 1–3 h |
| *(more models being added — see [`docs/models.md`](../docs/models.md))* | M3–M7 | mostly self-contained | minutes–tens of min |

## Honesty / provenance

The data is real and licensed CC BY 4.0 (see [`DATA_LICENSES.md`](../DATA_LICENSES.md)).
The notebooks never fabricate a number — every metric is computed from your run.
If a cell says a value is a **placeholder**, it means exactly that. Results you
return are credited to you and logged in [`AI_LOG.md`](../AI_LOG.md) for the
submission's AI-use disclosure.
