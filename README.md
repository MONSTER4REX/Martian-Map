# 🛰️ Marswalk Planner

**Layered, uncertainty-aware EVA route planning for a Martian AOI.**

Our team's entry for the **2026 NASA Space Apps Challenge** —
*"Interplanetary Survival Guide: Martian Map"* (hackathon Nov 14–15, 2026).

> **The claim we set out to prove:** on a real Jezero site, the full
> mission-constrained route differs from the shortest and slope-only routes
> for *defensible, explainable* reasons — with every assumption and every
> source of uncertainty shown.

Marswalk fuses multi-mission Mars data into a co-registered, uncertainty-aware
site model and plans human EVA / pressurized-rover routes under **hard
constraints** (walk-back, consumables, slope, hazards) and **soft objectives**
(time, energy, risk, science value, comms coverage).

---

## ✨ What it does

- **Three routes, one destination.** Shortest (3-D distance), slope-aware, and
  full mission-constrained — compared on length, time, energy, hazard exposure,
  walk-back margin, and science value, *with the reason each route is rejected.*
- **Walk-back as a hard constraint.** At every node and time, time-to-nearest
  safe haven ≤ remaining consumables − reserve, checked across contingencies
  (nominal, **rover-loss**, suit degradation).
- **Survival envelope.** No-go / death zones that collapse under a rover-loss
  contingency — the hero visual, with an interactive click-a-point console
  (SAFE / FATAL, live).
- **Multi-objective planning.** Exact Pareto fronts (NAMOA\*) on a corridor,
  validated against Dijkstra.
- **Science under walk-back.** Time-windowed orienteering: maximize science
  value while preserving return reserves (exact subset DP).
- **Rock-hazard routing.** A detector → planner link so rock-aware routes avoid
  rubble fields that slope-only routing would accept.
- **Route-stability map.** How often the recommendation changes under DEM-error
  and parameter perturbation — and which assumption drives it.

---

## 🧭 Design principles

1. **Planner first, ML only where it earns its place.** A feature ships only if
   tests pass *and* an ablation shows it measurably changes route decisions.
2. **Every layer is declared** `hard exclusion` / `objective cost` /
   `informational`. **Missing data is *unknown*, never *safe*.**
3. **Never present an unvalidated risk score as a failure probability.**
4. **Nothing network- or GPU-dependent is "done"** until an owner runs it on a
   real machine and reports back. Such code is marked **UNTESTED ON REAL DATA**.
5. **Never fabricate** data, metrics, citations, or links. The sandbox used to
   build this has no access to AWS / PDS / ODE / Zenodo and no GPU; the full
   offline test suite runs on synthetic fixtures.

---

## 🚀 Quickstart

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest          # full offline suite
ruff check .
```

**End-to-end mission report** (drives the real pipeline — three-route
comparison + walk-back flip across contingencies + route-stability figure):

```bash
marswalk mission                       # default ridge terrain, human-readable
marswalk mission --json                # machine-readable
marswalk mission --terrain random --seed 3
marswalk mission --start 0,0 --goal 39,0 --base 0,0 --rover 39,0
```

**Three-route demo in Python:**

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

**API + web map:**

```bash
pip install -e ".[api]"
uvicorn marswalk.api.app:app --reload      # POST /routes, POST /probe, GET /health
# then open web/index.html (map) or web/demo/index.html (survival console)
```

---

## 📂 Repository layout

```
src/marswalk/
  planner/      grid, connectivity, slope, cost, A*, layers, routes,
                mission (walk-back), multiobjective (NAMOA*), orienteering, hazard
  physics/      locomotion, consumables, walk-back, thermal, radiation, slip, survival
  geometry/     SPICE visibility / illumination / comms coverage
  science/      rock size-frequency fit, science-value raster
  data/         DEM derivatives, DTM QA, DEM error model, manifest, I/O
  ml/           rock detector, traversability, annotation, spatial splits, metrics
  validation/   route metrics, route stability, ablation harness
  terrain/      synthetic terrain for offline tests/demos
  api/          FastAPI service (routes, probe) + pydantic schemas
tests/          grid · slope · routes · A*-vs-NetworkX oracle · mission · more
web/            React map, survival console, pitch page, OpenAPI
notebooks/      training notebooks (rock detector, AI4Mars terrain, …) — run owner-side
docs/           per-module READMEs + HISTORY.md
scripts/        offline demo/export/figure-render scripts
```

Module docs live in [`docs/`](docs/) · data provenance & licenses in
[`DATA_LICENSES.md`](DATA_LICENSES.md) · full spec in
[`MARSWALK_CONTEXT.md`](MARSWALK_CONTEXT.md) · assumptions in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md).

---

## 🧱 Build order (planner before ML)

1. ✅ 3-D shortest path — checked against an independent implementation (NetworkX)
2. ✅ Slope-aware cost
3. ✅ Hard exclusions (slope, cross-slope, unknown = blocked)
4. ✅ Mission constraints + walk-back
5. ✅ Multi-objective NAMOA\* on a corridor — exact front, validated vs Dijkstra
6. ✅ Time-windowed orienteering — science under walk-back

ML (rock detector) is gated: it ships only after the baseline planner passes
and an ablation shows it changes route decisions.

---

## 🛰️ Data sources

Marswalk is designed to fuse HiRISE / CTX DTMs, THEMIS thermal inertia, the
Jezero geologic map, CRISM mineralogy, MEDA/REMS weather, SPICE geometry, and
Perseverance traverse data. **Input datasets keep their own terms and are not
redistributed** — see [`DATA_LICENSES.md`](DATA_LICENSES.md) (HRSC is CC BY-SA;
MCD is non-commercial; some rover-view sets have unspecified licenses). Coarse
layers (thermal inertia ~100 m, GRS hundreds of km) are **never** presented as
EVA-scale detail.

---

## ⚠️ Status & honesty notes

- The planner build (steps 1–6), EVA physics, survival envelope, validation
  harness, API, and web demo all run **offline on synthetic terrain**.
- All physics / threshold parameters are **placeholders** (see
  [`ASSUMPTIONS.md`](ASSUMPTIONS.md)) — the CLI demonstrates the *decision
  machinery*, not calibrated Mars values.
- Real-data co-registration, SPICE geometry, and model training are run on team
  machines and reported back; until then they're marked **UNTESTED ON REAL DATA**.
- AI assistance was used in development and is logged in [`AI_LOG.md`](AI_LOG.md).

---

## 📜 License

**Code: MIT** ([`LICENSE`](LICENSE)). Input datasets are not redistributed and
retain their original terms — see [`DATA_LICENSES.md`](DATA_LICENSES.md).

---

*Built for the 2026 NASA Space Apps Challenge · "Interplanetary Survival Guide: Martian Map"*
