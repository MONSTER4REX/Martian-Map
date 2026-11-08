# AI authorship log (for submission disclosure)

Kept so the team can disclose AI-assisted work accurately (Open item A1 —
event rules not yet confirmed). Each row: what the AI produced, who reviewed /
owns it, and what remains human-verified. Owners: update "Reviewed by" after
you read and run the code.

> Convention: the AI (Claude, Opus 4.8) writes code, tests and docs offline on
> synthetic fixtures. Owners run on real data, review, and own their module for
> judging Q&A. Nothing network/GPU-dependent is "done" until an owner pastes
> back real results.

| Date | Author | Artifact | Reviewed by | Notes |
|---|---|---|---|---|
| 2026-11-08 | Claude (Opus 4.8) | Repo skeleton (pyproject, ruff, pytest, CI, licenses, docs) | _pending_ | — |
| 2026-11-08 | Claude (Opus 4.8) | P3 planner baseline: `grid`, `connectivity`, `slope`, `cost`, `astar`, `layers`, `routes`; synthetic terrain | _pending (P3)_ | Offline, synthetic terrain only. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | Test suite: A* vs NetworkX oracle (randomized), grid, slope, routes | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | P4 EVA physics: `locomotion`, `consumables`, `walkback` (nominal / rover-loss / suit-degraded) | _pending (P4)_ | Offline, synthetic terrain. ALL parameters are placeholders. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | P4 tests (locomotion, consumables, walk-back incl. rover-loss decision flip) | _pending (P4/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/mission.py`: walk-back wired as a hard planner constraint (time-objective A* + prune); route-level scenario flip | _pending (P3/P4)_ | Offline. Satisfies §3 gate. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | Mission tests (soundness self-check, scenario flip, infeasible goal/start) + docs/mission.md | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/multiobjective.py`: exact Pareto (NAMOA*-style Martins) on a corridor + `corridor_mask`; docs/multiobjective.md | _pending (P3)_ | Offline. Extremes validated vs NetworkX. |
| 2026-11-08 | Claude (Opus 4.8) | Multi-objective tests (dominance, non-dominated front, per-objective extremes vs Dijkstra, corridor) | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | `planner/orienteering.py`: time-windowed orienteering (exact Held-Karp subset DP) under walk-back; docs/orienteering.md | _pending (P3)_ | Offline. |
| 2026-11-08 | Claude (Opus 4.8) | Orienteering tests (budget subset, time window, walk-back/unreachable exclusion, determinism) | _pending (P3/P6)_ | Runs offline on synthetic fixtures. |
| 2026-11-08 | Claude (Opus 4.8) | P1 `data/derivatives.py` (slope/aspect/hillshade/curvature) + `data/qa.py` (C1 DTM QA) + tests + docs/data.md | _pending (P1)_ | Offline on synthetic DEMs. |
| 2026-11-08 | Claude (Opus 4.8) | P1 `data/error_model.py` (C2 correlated-noise ensembles, empirical variogram, exponential fit) + tests | _pending (P1/P6)_ | Offline. Fit params must come from real residuals. |
| 2026-11-08 | Claude (Opus 4.8) | P6 `validation/` metrics (jaccard, 3-D length, hazard, discrete Fréchet), route stability, ablation harness + tests + docs/validation.md | _pending (P6)_ | Offline on synthetic terrain + DEM ensembles. |
| 2026-11-08 | Claude (Opus 4.8) | P1 `data/manifest.py` (STAC-ish layer records + license/res/uncertainty) + `data/io.py` (COG window reader, rasterio-guarded) + tests | _pending (P1)_ | Manifest tested offline; COG read UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | C8 `geometry/` (LOS visibility, illumination, comms coverage) + synthetic ephemeris + guarded `spice.py` + tests + docs/geometry.md | _pending (P1)_ | Geometry tested offline; real SPICE UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | C7 `science/size_frequency.py` (exponential SFD fit) + C9 `science/value_raster.py` (weighted value raster + top_targets) + tests + docs/science.md | _pending (P2/P6)_ | Offline, input-agnostic. Golombek-Rapp CFA calibration still owed. |
| 2026-11-08 | Claude (Opus 4.8) | P2 `ml/` annotation schema + spatial splits + detection metrics (F1 by size bin) + guarded `train.py` + scripts/train_rock_detector.py + tests + docs/ml.md | _pending (P2)_ | Schema/splits/metrics tested offline; training UNTESTED ON REAL DATA (needs torch+GPU). |
| 2026-11-08 | Claude (Opus 4.8) | P3/P5 `api/` (pure service + FastAPI /routes /probe /health + OpenAPI) + `web/index.html` (React map, probe) + export script + tests + docs/api.md | _pending (P3/P5)_ | Tested offline incl. FastAPI TestClient; verified rendering in-browser. |
| 2026-11-08 | Claude (Opus 4.8) | P4 C3 `thermal.py` / C4 `radiation.py` / C6 `slip.py` screening (placeholders + validity notes) + tests + docs/screening.md | _pending (P4)_ | Offline, structural tests only. PLACEHOLDER models; validate vs MEDA / negative rover cases. |
| 2026-11-08 | Claude (Opus 4.8) | `cli.py` (`marswalk mission` end-to-end demo: 3-route compare + walk-back flip + route stability) + `[project.scripts]` + tests + docs/cli.md | _pending (P3/P5)_ | Offline. Drives the real modules; no hand-authored numbers. UNTESTED ON REAL DATA. |
| 2026-11-08 | Claude (Opus 4.8) | `scripts/render_mission_figure.py` (3-panel demo PNG) + `viz` extra + smoke test + docs/mission_figure.png | _pending (P5)_ | Offline. Rendered from the real pipeline. Figure is a synthetic-terrain demo; PLACEHOLDER params. |
| 2026-11-08 | Claude (Opus 4.8) | `notebooks/train_rock_detector.ipynb` — **real BOULDERING data** (Zenodo 8171052, CC BY 4.0): MD5-verified download, Detectron2-JSON loader, authors' split, Faster R-CNN, F1-by-size; `nb` extra; torchvision/pillow in `ml` | _pending (P2)_ | Verified against the real 265 MB download (CPU): JSON load (953 patches / 39k boxes; Mars subset 49/6.8k), split, train+eval step all run. Trained result owner-side (GPU). Gated by §3 ablation. |
| 2026-11-09 | Claude (Opus 4.8) | `notebooks/train_ai4mars_terrain.ipynb` — **real AI4Mars data** (Zenodo 15995036, CC BY 4.0): DeepLabV3 terrain segmentation, NAV 4-class key (verified from `label_keys.json`), MSL NavCam, per-class IoU + big-rock IoU | _pending (P2)_ | Structure + classes verified by range-reading the 16 GB archive; load→train→IoU verified on real extracted MSL NavCam patches (8 train/4 test, CPU). Full run owner-side (16 GB + GPU). Rover-view, not orbital. Gated by §3. |
| 2026-11-09 | Claude (Opus 4.8) | Rock-hazard layer: `planner/hazard.py` (field + exposure), `planner/hazard_routes.py` (hazard-aware A*), `validation/ablation.py::rock_ablation` (§3 gate) + 7 tests + `scripts/rock_ablation_demo.py` + `examples/bouldering_ESP_017355_2260.json` + docs/hazard.md | _pending (P3/P6)_ | Offline, tested. The detector→planner link + the §3 gate. Demo runs the gate on a REAL Mars boulder field (461 boulders, HiRISE ESP_017355_2260): rock-aware cut exposure 98% and changed the route. Hazard magnitudes are PLACEHOLDERS; AOI is a demo scale. |
| 2026-11-09 | Claude (Opus 4.8) | Survival envelope (hero): `physics/survival.py` (reach field, no-go/death zones, scenario collapse) + 6 tests + `scripts/survival_demo.py` + docs/survival.md + docs/survival_envelope.png | _pending (P4)_ | Offline, tested. Where you can go and still walk back alive; collapses 80%→47%→7% under rover-loss/suit-degraded. Built on the real walk-back engine. PLACEHOLDER life-support; synthetic terrain (real DEM drops in). |

## How to keep this honest

- Add a row when the AI produces or substantially rewrites a module.
- When an owner rewrites AI code by hand, note it (author changes to that owner).
- Keep commit messages descriptive; this table plus `git log` is the disclosure.
