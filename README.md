# Marswalk Planner

Layered, uncertainty-aware **EVA route planning** for a Martian AOI. Our team's
entry for the **2026 NASA Space Apps Challenge**, "Interplanetary Survival
Guide: Martian Map" (hackathon Nov 14–15, 2026).

> **Core claim to prove:** on a real Jezero site, the full mission-constrained
> route differs from the shortest and slope-only routes for defensible,
> explainable reasons, with assumptions and uncertainty shown.

Spec and decisions: [`MARSWALK_CONTEXT.md`](MARSWALK_CONTEXT.md) (source of
truth) and [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## Status (2026-11-08)

| Area | State |
|---|---|
| Repo skeleton, CI, licenses | ✅ in place |
| **P3 planner baseline** (grid, A*, slope, 3-route comparison) | ✅ offline; **UNTESTED ON REAL DATA** |
| A* validated vs independent oracle (NetworkX) | ✅ randomized suite |
| **P4 EVA physics** (locomotion, consumables, **walk-back** gate) | ✅ offline, placeholder params; **UNTESTED ON REAL DATA** |
| **Walk-back wired into planner** (mission constraint, step 4) | ✅ offline; gate satisfied → step 5 unlocked |
| **Survival envelope** (no-go/death zones; collapses under rover-loss) | ✅ offline; the hero visual ([docs/survival.md](docs/survival.md)) |
| **Interactive survival console** (click a point → SAFE/FATAL, live) | ✅ offline; the judge-facing demo ([docs/demo.md](docs/demo.md)) |
| **Step 5: multi-objective NAMOA\*** (Pareto on a corridor) | ✅ offline; exact front, validated vs Dijkstra |
| **Step 6: time-windowed orienteering** (science under walk-back) | ✅ offline; exact subset DP |
| **P1 data (offline)** — DEM derivatives, DTM QA (C1), DEM error ensembles (C2) | ✅ offline; network parts pending |
| **P6 validation** — route metrics, route stability, ablation harness | ✅ offline |
| **Rock-hazard layer + §3 gate** — detector→planner link, rock-aware routing | ✅ offline; gate demo on **real** Mars boulders ([docs/hazard.md](docs/hazard.md)) |
| **C8 geometry** — visibility / illumination / comms coverage | ✅ offline; real SPICE pending |
| **C7/C9 science** — rock size-frequency + value raster | ✅ offline |
| **P2 ML** — annotation, spatial splits, detection metrics | ✅ offline; training pending (GPU) |
| **Training notebook** — Faster R-CNN on **real BOULDERING data** (CC BY 4.0) | ✅ verified on real download (CPU); full GPU run owner-side |
| **AI4Mars notebook** — DeepLabV3 terrain segmentation on **real AI4Mars** (CC BY 4.0) | ✅ verified on real patches (CPU); full GPU run owner-side |
| **Model roadmap** — restored PRD v0.1 scope (6 trained + 1 optional); run-and-return notebooks | 🔄 M1 training on the dev GPU now; M3–M7 being built ([docs/models.md](docs/models.md), [notebooks/README.md](notebooks/README.md)) |
| **P3/P5 API + web** — FastAPI (/routes /probe) + React map | ✅ offline; verified in-browser |
| **`marswalk` CLI** — end-to-end mission report (3-route + walk-back + stability) | ✅ offline; drives real modules |
| **Demo figure** — 3-panel render (routes + walk-back field + stability map) | ✅ offline ([docs/cli.md](docs/cli.md)) |
| **C3/C4/C6 screening** — thermal / radiation / slip (placeholders) | ✅ offline; validate vs MEDA / rover cases |
| Whole suite | ✅ 208 tests pass (offline) |
| Remaining | ⏳ real-data / GPU runs on owner machines; model validation |

Nothing network- or GPU-dependent is "done" until an owner runs it on their
machine and pastes back results (the sandbox has no AWS/PDS/ODE/Zenodo access
and no GPU).

### Assumptions baked in (the three blocking questions, defaults taken)

1. **No pre-existing repo** — this is the starting skeleton.
2. **Event rules UNCONFIRMED** — pre-event-work and AI-use/disclosure rules not
   yet checked against the 2026 Participant Terms. Until confirmed, all code is
   prep, AI authorship is logged in [`AI_LOG.md`](AI_LOG.md), and AI use will be
   disclosed. (Lead to confirm.)
3. **AOI = Jezero**, roles/machines per `MARSWALK_CONTEXT.md` §3–4, not yet
   re-confirmed by a coverage audit.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest            # 208 tests, offline
ruff check .
```

End-to-end mission demo (drives the real pipeline):

```bash
marswalk mission            # or: PYTHONPATH=src python -m marswalk.cli mission
```

Three-route demo:

```python
from marswalk.terrain.synthetic import ridge_world
from marswalk.planner.routes import compare_routes

grid = ridge_world(40, 40, cellsize=2.0, ridge_height_m=2.0,
                   ridge_half_width=1, gap_col=20)
cmp = compare_routes(grid, start=(0, 0), goal=(39, 0))
print(cmp.summary)
# -> Shortest route is NOT mission-feasible (2 hard-exclusion edge(s)); the
#    mission route detours around them. Mission route is +37.4% vs shortest.
```

## Layout

```
src/marswalk/
  planner/      grid, connectivity, slope, cost, astar, layers, routes
  terrain/      synthetic terrain for offline tests/demos
tests/          grid, slope, routes, A*-vs-NetworkX oracle
docs/           planner.md (module README)
data/           gitignored — never commit rasters (DVC / external SSD)
```

Module docs: [`docs/planner.md`](docs/planner.md).
Data provenance & licenses: [`DATA_LICENSES.md`](DATA_LICENSES.md).

## Build order (from the spec)

Planner first, validated at each step, **before** any ML:
1. ✅ 3-D shortest path (checked vs independent impl)
2. ✅ slope-aware cost
3. ✅ hard exclusions (slope, cross-slope, unknown=blocked)
4. ✅ mission constraints + walk-back (P4) — walk-back is now a hard planner
   constraint ([docs/physics.md](docs/physics.md), [docs/mission.md](docs/mission.md)).
   Gate satisfied.
5. ✅ multi-objective NAMOA* on a corridor ([docs/multiobjective.md](docs/multiobjective.md))
6. ✅ time-windowed orienteering ([docs/orienteering.md](docs/orienteering.md))

The planner build order (steps 1–6) is complete offline; remaining work is the
data layer (P1), validation (P6), geometry/science rasters, ML, and the API +
frontend.

ML (rock detector) only after the baseline planner passes and an ablation shows
it changes route decisions (`MARSWALK_CONTEXT.md` §3 gates).

## License

Code: MIT ([`LICENSE`](LICENSE)). Input datasets keep their own terms and are
not redistributed — see [`DATA_LICENSES.md`](DATA_LICENSES.md) (HRSC is CC BY-SA;
MCD is non-commercial; AI4Mars license unspecified).
