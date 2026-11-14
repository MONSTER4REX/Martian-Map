# Marswalk — development history

A persistent, human-readable record of how this project was built, the decisions
behind it, and where it stands — so the full story survives outside any single
chat session. Kept in git; update it as the project moves.

- **Repo:** github.com/MONSTER4REX/Martian-Map · **Branch:** `main`
- **Event:** NASA Space Apps 2026, "Interplanetary Survival Guide: Martian Map"
  (hackathon Nov 14–15, 2026)
- **Build window covered:** 2026-11-08 → 2026-11-14
- **Spec / source of truth:** [`MARSWALK_CONTEXT.md`](../MARSWALK_CONTEXT.md),
  assumptions in [`ASSUMPTIONS.md`](../ASSUMPTIONS.md), AI-authorship disclosure
  in [`AI_LOG.md`](../AI_LOG.md).

---

## 1. What Marswalk is

An open-source **EVA route planner reframed as a survival system**. Most Mars
map tools optimize distance or energy; Marswalk gates on the question that keeps
a crew alive: *at every step, can you still walk back to a safe haven before
consumables run out?* It turns that **walk-back guarantee** into a map of where
you can and cannot safely stand, and plans routes that never cross the line.

**The hero capability:** the *survival envelope* — the survivable world shrinks
**80% → 47% → 7%** as the rover is lost and then the suit degrades.

---

## 2. Principles we worked under (non-negotiable)

1. **Never fabricate.** No invented data values, metrics, citations, or links.
   Anything not yet run on real data is marked **UNTESTED**; placeholders are
   labelled as placeholders. This scientific honesty is treated as a *feature*.
2. **Build the planner first, ML only where it changes decisions** (gated by an
   ablation win) — per the spec's build order.
3. **Sandbox limits:** the dev/CI sandbox has no AWS/PDS/ODE/Zenodo access and
   (originally assumed) no GPU. Network/GPU work is owner-side and marked so;
   tests run offline on synthetic fixtures.
4. **Commit attribution:** commits show **only the user's username (MONSTER4REX)**
   — no Co-Authored-By / secondary author. (Set 2026-11-09.)

---

## 3. Build timeline

### Phase 1 — Planner + physics (2026-11-08)
Planner built step-by-step, each validated before the next:

| Commit | What |
|---|---|
| `c4a29b4` | Repo skeleton + P3 planner baseline (grid, A*, slope, 3-route compare) |
| `2b3ea14` | P4 EVA physics: locomotion, consumables, **walk-back gate** |
| `7b8e3a9` | Walk-back wired as a **hard** planner constraint (step 4) |
| `8e34561` | Step 5: multi-objective NAMOA\*-style Pareto routing |
| `b8f1f77` | Step 6: time-windowed orienteering (science under walk-back) |

A* validated against an independent NetworkX oracle (randomized suite).

### Phase 2 — Data, geometry, science, validation, API (2026-11-08)
| Commit | What |
|---|---|
| `4067ac3` / `cb5c1cc` | P1 DEM derivatives + DTM QA (C1); DEM error ensembles (C2) |
| `a6699b1` | P6 validation: route metrics, route stability, ablation harness |
| `47e0246` | P1 data manifest + windowed COG reader (COG read UNTESTED ON REAL DATA) |
| `2a6f3da` | C8 geometry: visibility / illumination / comms (synthetic ephemeris) |
| `9190be4` | C7 rock size-frequency fit + C9 science-value raster |
| `e52f937` | P2 rock-detector schema, spatial splits, detection metrics |
| `7285d4a` | P3/P5 planner API (FastAPI + OpenAPI) + minimal React map |
| `42be1b2` | C3 thermal / C4 radiation / C6 slip screening (placeholders) |
| `e3d0253` / `28e069d` | `marswalk` CLI end-to-end demo + 3-panel figure |

### Phase 3 — Real NASA data + the winning pivot (2026-11-08/09)
| Commit | What |
|---|---|
| `9f3b0d2` / `a154463` | Rock-detector notebook on **real BOULDERING** (Zenodo 8171052, CC BY 4.0) — no synthetic data |
| `b946415` | AI4Mars terrain-segmentation notebook on **real AI4Mars** (Zenodo 15995036, CC BY 4.0) |
| `1869e21` | Rock-hazard layer + **§3 ablation gate** — demo on a real Mars boulder field (461 boulders, HiRISE ESP_017355_2260): rock-aware routing cut rock exposure **−98%** and changed the route |
| `2c305d0` | **Survival envelope** — the hero visual (80→47→7 collapse) |
| `0379209` / `44e8a5e` | **Pitch page** (survival-first landing), later linked to the live demo |
| `7e8b399` | **Interactive survival console** — click a point on Jezero → SAFE/FATAL verdict, live, driven by the real walk-back engine |

**Strategic reframe (why we pivoted):** judges reward story + a demo a
non-expert gets in 2 minutes + a "wow" moment, not code depth. So Marswalk was
reframed from an abstract route planner into a **survival system** with the
walk-back guarantee as the unique hook. Two shareable artifacts were published
(private; share via each page's Share menu):
- Pitch page: `claude.ai/artifact/CMj6ksFfSBCnYYmYTsRJCd`
- Interactive demo: `claude.ai/artifact/BBsEB6amC1y74boN4ryrA9`

### Phase 4 — The six models restored + trained (2026-11-09 → 11-12)
The lead restored the **six-model scope** from PRD v0.1 (v0.2 had trimmed it to
2). All were built as tested modules + friend-runnable notebooks.

| Commit | What |
|---|---|
| `5c40c91` | Restore 6-model scope + friend run-and-return workflow (`docs/models.md`, `notebooks/README.md`) |
| `e129050` / `c794e26` | **M7** learned planner heuristic + **M1** GPU training pipeline (verified on the GTX 1650) |
| `d247ea2` | **M6** DTM-artifact detector |
| `7408f1d` | **M5** thermal emulator |
| `4a7fd1d` | **M3** traversability (PU-learning) + **M4** dose regression |
| `111feab` | M1 notebook self-installing + per-laptop assignment table |
| `c2887da` | CI lint fixes + **M1 trained 8 epochs** on the GTX 1650 |
| `4cdb790` | Notebooks print absolute output paths; real metrics logged |

---

## 4. The models (M1–M7) and real results

See [`docs/models.md`](models.md) for full status. Verified numbers (from actual
runs, never hand-entered):

| # | Model | Data | Result |
|---|---|---|---|
| **M1** | Rock detector (Faster R-CNN) | BOULDERING (real) | **F1 0.48** (P 0.65 / R 0.38), 8 epochs on GTX 1650, 2.08 GB VRAM; weights saved |
| **M2** | Surface classifier (DeepLabV3) | AI4Mars (real, 16 GB) | notebook ready; full run pending |
| **M3** | Traversability (PU-learning) | rover path + terrain | **ROC-AUC ~1.000** (held-out), CPU |
| **M4** | Dose regression | C4 model surrogate | **1.0% median error**, CPU |
| **M5** | Thermal emulator | C3 model surrogate | **MAE 0.25–0.26 K**, CPU |
| **M6** | DTM-artifact CNN | injected box-step/CCD-seam | **99.7% acc / F1 ~1.0** (unseen terrain), CPU |
| **M7** | Learned heuristic (optional) | planner-generated | beats straight-line baseline, ~99% admissible |

**Honesty flags:** M1/M2 train on real public labels. M5/M6/M7 are
self-contained / real-mechanism (M5/M4 emulate C3/C4 **placeholder** physics
models; M6 uses injected-artifact weak supervision). M3's rover traverse is
**synthetic** until real Perseverance/Curiosity waypoints are wired. No model's
number is reported that wasn't computed by a run.

---

## 5. How to run / operate

**Tests (offline):** `pytest` → **220 pass**, 1 skipped. `ruff check .` clean.

**GPU training env (this dev laptop, GTX 1650 4 GB):** a separate `.venv-gpu`
holds CUDA PyTorch (kept off the near-full C: drive; build caches on D:).
`scripts/train_rock_detector.py` trains M1; it downloads BOULDERING (~265 MB,
md5-verified) to `data/bouldering/` (gitignored).

**Friend notebooks (run-and-return):** 7 notebooks in `notebooks/`. The light 5
(M3–M7) self-install the repo (`pip install git+…`) → upload to Colab, GPU
optional, **Run all**; M1 self-installs too; M2 needs only torch + its download.
Each writes `<model>_report.json` + `<model>_samples.png` and prints their
**absolute paths**; runners send both back. Per-laptop assignment and prereqs
are in [`notebooks/README.md`](../notebooks/README.md).

---

## 6. Open items

**Known issue — CI red:** GitHub Actions can't start because the account is
**locked over a billing issue** ("the job was not started because your account
is locked due to a billing issue"). The code is lint-clean and green locally;
once billing is resolved and jobs re-run, checks should pass. Only the account
owner can clear this (GitHub → Settings → Billing).

**Owner / team-side (documented, turnkey):**
- Drop in a **real Jezero DEM** (HiRISE/CTX) — planner/survival/demo currently
  run on synthetic terrain; the real DEM makes it a real site. *Highest-value
  next step.*
- Run **M2** (16 GB) and distribute **M3–M7** to the team laptops for real
  numbers; a longer **M1** run on a 6 GB card for better recall.
- Replace placeholders with **cited values**: life-support (O₂/consumption),
  thermal (MEDA), radiation (RAD), and real GSD for metre-scale rock sizes.
- **P6 validation** vs real rover traverses (Perseverance held-out), MEDA, etc.

**Submission polish:**
- 2-minute pitch **narration/video script** (offered; not yet written).
- Confirm **event rules** (pre-event-work + AI-disclosure vs 2026 Participant
  Terms) — a lead decision, flagged in `AI_LOG.md`.

---

## 7. File map

```
src/marswalk/
  planner/     grid, A*, slope, cost, layers, routes, mission, multiobjective, orienteering
  physics/     locomotion, consumables, walkback, survival (hero), thermal, radiation, slip
  planner/hazard*.py   rock-hazard field + hazard-aware routing
  data/        derivatives, qa, error_model, manifest, io (COG)
  geometry/    visibility, illumination, comms, spice (guarded)
  science/     size_frequency, value_raster
  ml/          annotation, splits, metrics_det, + heuristic, dtm_artifacts,
               thermal_emulator, traversability, dose_regression
  api/         FastAPI service (/routes /probe /health)
scripts/       survival_demo, rock_ablation_demo, export_jezero_demo,
               train_rock_detector, render_mission_figure, make_ml_notebooks
notebooks/     7 run-and-return training notebooks + README
web/pitch/     survival-first pitch page
web/demo/      interactive survival console (+ survival_data.{json,js})
docs/          module docs incl. survival.md, demo.md, models.md, this file
tests/         220 offline tests (synthetic fixtures / planner oracles)
```

---

*Maintenance: add a dated row to the timeline and update §4/§6 whenever a module
lands or a real result comes back. This file is the project's memory.*
