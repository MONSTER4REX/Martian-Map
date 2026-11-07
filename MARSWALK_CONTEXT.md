# Marswalk Planner: Project Context and Chat History (handoff brief)

Compiled Nov 8, 2026 from the working conversation between the team lead (Rudresh) and Claude. This file is self-contained: the agent reading it has not seen the chat. PRD v0.2 is the spec; section 3 summarizes it. If this file and an older plan disagree, this file wins.

## 1. Situation

- **Event:** 2026 NASA Space Apps Challenge, challenge "Interplanetary Survival Guide: Martian Map". Ask (paraphrased): build a layered, integrated view of a Martian location or route that fuses data from multiple NASA missions and helps a human explorer plan and carry out a Marswalk.
- **Dates (event site, checked Oct 11):** hackathon Nov 14-15, 2026. Submission closes Sun Nov 15, 11:59 PM local. Submission and Judging guides publish Nov 13. Judging starts Nov 16. **Today is Nov 8, so kickoff is 6 days away.** Full challenge statements were due Oct 28 and may have changed details; the lead should re-check.
- **Team:** six people (P1-P6), B.E. Data Science students. Lead: Rudresh.
- **Not confirmed in the chat:** (a) whether work done before Nov 14 is allowed, (b) the rules on AI-assisted work and disclosure. The lead must check the Participant Terms or ask the local event lead. Until confirmed, treat code as prep, keep a log of what AI wrote, and plan to disclose AI use.
- **Current build status:** unknown to this brief. Ask the lead for the repo state before duplicating work.

## 2. Decision history (what was decided and why)

1. **Challenge choice.** 13 challenges were ranked by real NASA workflow gap, defensibility and crowding. MODIS/VIIRS harmonization, CLPS lunar browser, trend detective and NISAR were considered. NISAR was high data-risk (provisional L-band data, only about 8-9 acquisitions per location by autumn). The team chose **Martian Map**: unique, plausibly useful to NASA, resume-worthy.
2. **First plan (PRD v0.1).** An EVA route planner with six trained models plus one optional.
3. **Independent review (ChatGPT) was mostly adopted.**
   - Planner first; ML only where an ablation shows it changes decisions.
   - Dropped: PU-learning traversability from the rover path, dose regression, thermal emulator, DTM-artifact CNN, learned planner heuristic.
   - Rover-traverse agreement is consistency evidence only, not proof of safety or the single correct route.
   - Do not resample coarse data into false detail; missing data is unknown, not safe.
   - AOI shrunk; spatially disjoint train/test splits; SAM labels are proposals, not ground truth.
4. **Additional points from Claude.**
   - The event is two days long, so the schedule is tight.
   - Negative validation cases exist at other sites: Spirit's Troy embedment, Opportunity's Purgatory Dune, Curiosity's slip-limited ripple crossings. Check coverage first.
   - Roles were rebalanced (P6 was overloaded).
   - Hardware: Macs can train small nets via MPS; Kaggle/Colab free GPUs for bigger runs; stream COGs from AWS instead of downloading.
   - Formal ship/cut gates; decision-flip demos; a route-stability map as the headline uncertainty figure.
5. **PRD v0.2** was written (planner-first, owners, datasets per section, gates, schedule). Per-role link lists were produced; the RTX 4050 laptop's list (rock detector) was turned into a Word file.
6. **Workflow decision.** Claude writes code and tests; owners run them on real data and report back, review their own modules, and own them for judging Q&A. One shared repo with CI. Claude's sandbox cannot reach AWS, Zenodo, PDS or ODE and has no GPU, so real-data and training behavior is only verified on the team's machines.

## 3. Spec summary (PRD v0.2)

**Product:** an open-source EVA decision-support platform. It fuses multi-mission Mars data into a co-registered, uncertainty-aware site model and plans human EVA / pressurized-rover routes under hard constraints (walk-back, consumables, thermal, comms) and soft objectives (time, energy, risk, science, comms coverage, dose).

**Core claim to prove:** on a real Jezero site, the full mission-constrained route differs from the shortest and slope-only routes for defensible, explainable reasons, with assumptions and uncertainty shown.

**Approach**
1. Planner first: real terrain, then shortest path, slope-aware, hard exclusions, mission constraints, multi-objective. Validate each step before the next.
2. Three routes, one destination: shortest, slope-aware, full mission-constrained, with distance, time, energy, hazard exposure, science value and the reason a route is rejected.
3. Every layer is declared **hard exclusion**, **objective cost** or **informational**. Missing data = unknown.
4. Features ship only if tests pass and an ablation shows a measurable change.
5. Never present an unvalidated risk score as a failure probability.
6. Decision-flip demos: (a) the shortest route violates walk-back under a rover-loss contingency; (b) best EVA start time shifts with dust/thermal; (c) slope-only accepts a rubble field that rock-aware routing avoids.
7. Route-stability map: how often the recommendation changes under DEM and parameter perturbation, and which assumption drives it.
8. Science under walk-back: user picks targets; planner maximizes science while preserving return reserves.
9. What-if scenarios: start time, consumables, dust, comms. Recompute and state which constraint changed the decision.

**AOI:** Jezero, chosen by coverage audit. Core ~5x5 km at ~2 m (~6M cells); context ~20 km at 10 m (~4M cells). Multi-objective search on a corridor mask only. Each layer keeps native resolution, an effective-resolution tag and uncertainty.

**Architecture**
- Storage: COG + Zarr, STAC; big files via DVC or external SSD, never plain Git.
- Access: windowed COG reads from AWS Open Data (rasterio `/vsis3/`, anonymous; verify).
- Processing: GDAL, rasterio, xarray + dask, ISIS3 only for PDS-only products, SpiceyPy.
- Planner: Python + Numba first; Rust/C++ only if profiling demands.
- Services: FastAPI + OpenAPI (schema co-owned by P3 and P5).
- Frontend: React; MapLibre/deck.gl 2D; CesiumJS or Three.js for 3D (verify Mars terrain workflow early).
- Frames: IAU Mars body-fixed frame; local projection per AOI; units declared on every layer.

**Roles**

| Role | Owns |
|---|---|
| P1 Data/GIS (on the RTX 4050 laptop) | Acquisition, CRS, co-registration, DEM derivatives, quality masks, DTM QA rules, DEM error model, manifest, CI; after Phase 1, SPICE visibility and comms geometry |
| P2 Vision/ML | Rock detector (core), annotation, surface classifier (gated), rock size-frequency fit, science-value raster |
| P3 Planner | Graph, cost fields, A*, constraints, NAMOA*, orienteering, performance, API schema |
| P4 EVA physics | Locomotion energy, consumables, walk-back, thermal physics, radiation screening, slip proxy |
| P5 Frontend | 2D/3D map, layers, route comparison, timeline, scenarios, probe tool |
| P6 Validation | Baselines, Perseverance consistency tests, negative cases, ablations, route stability, docs, demo narrative |

The RTX 4050 laptop (6 GB) is the planned rock-detector training machine, so P1 and P2 share it.

**Trained models:** 1 core + 1 gated.
- **M1 Rock detector (core).** U-Net or Mask R-CNN with a small pretrained encoder, 256-512 px tiles, batch 2-4. Spatial holdout; precision/recall by size bin; aspirational F1 >= 0.8 for rocks >= 1.5 m. Gate: keep as a hazard objective only if rock-aware routes reduce rock exposure vs slope-only. Fallback: literature rock-abundance prior.
- **M2 Surface classifier (gated).** Start only after M1 v1 and the baseline planner. Weak labels from the USGS Jezero geologic map. Ships only if it improves held-out route decisions.

**Non-ML modules**
- C1 DTM QA rules (no-data, seams, discontinuities, slope spikes).
- C2 DEM error model (variogram-fit correlated noise ensembles).
- C3 Thermal physics (validate on a comparable quantity; ground and air temperature are not interchangeable).
- C4 Radiation screening (documented scalar with a stated validity range).
- C5 Locomotion, energy and consumables (literature parameters, baseline and conservative sets).
- C6 Slip proxy.
- C7 Rock size-frequency fit.
- C8 SPICE visibility, illumination and comms geometry.
- C9 Science value raster.
- C10 Ice/ISRU context (stretch, informational).
- C11 Dust context.

**Planner order:** (1) 3D-distance shortest path checked against an independent implementation; (2) slope-aware cost; (3) hard exclusions; (4) mission constraints and walk-back; (5) multi-objective NAMOA*-style search on a corridor; (6) time-windowed orienteering. Do not start (5) before (4) is verified.

**Walk-back:** at every node and time, time-to-nearest-safe-haven <= remaining consumables time - reserve, for nominal and contingency cases (rover loss, suit degradation).

**Metrics (initial):** planner matches an independent shortest-path implementation on a test suite; three-route comparison on length, time, energy, slope, rock exposure, walk-back margin, comms coverage, science value; hazard consistency on held-out Perseverance segments (reported with baselines and negative cases, never pass/fail alone); route stability; 5 km single-objective reroute <= 3 s; clean-install rebuild in one command.

**Gates:** rock-aware cost needs an ablation win; surface classifier needs a decision improvement; NAMOA* needs the A* baseline and constraints verified; thermal needs matched-quantity MEDA validation; radiation needs a stated validity range; science orienteering needs the walk-back check. A feature that fails its gate is cut or hidden.

**Validation (P6):** baselines on all route metrics; Perseverance held-out consistency (Frechet distance, hazard exposure); negative cases at other rover sites; rock-detector spatial holdout; thermal vs MEDA; SPICE geometry tests; planner property tests; ablations; route stability and sensitivity; second-AOI transfer with no retuning.

## 4. Hardware

| Machine | Use |
|---|---|
| HP Victus, RTX 4050 6 GB, i7-12650H, 16 GB RAM, ~268 GB free | Rock-detector training, ablations |
| ASUS Zephyrus G14, RTX 3050 Ti 4 GB, Ryzen 9 5900HS, 16 GB RAM, 1 TB SSD | Preprocessing, inference, surface-classifier trials |
| ASUS TUF A15, GTX 1650 4 GB, Ryzen 5, 16 GB RAM, 1 TB SSD | Preprocessing, inference, small experiments |
| MacBook Air M4 and M5, 16 GB each | GIS, planner dev, validation, MPS training fallback |
| MacBook Air M1, 8 GB, 256 GB | Docs, frontend, API testing; not a raster workstation |

4 GB training tips: pretrained small encoders (ResNet18/34, EfficientNet-B0), 256 px tiles, batch 2-4, gradient accumulation, `channels_last`, checkpointing if needed. AMP saves memory on the 1650 but gives little speedup (no tensor cores). Log `torch.cuda.max_memory_allocated()` and epoch time. Pre-cut tiles to disk once. Do not run training and horizon ray-march at the same time on one GPU.

## 5. Data links (collected from live searches in the chat)

Rows marked *root* were not opened; navigate from them.

**Terrain and imagery**
- HiRISE DTM catalog: https://www.uahirise.org/dtm/
- HiRISE DTM (PDS): https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MRO-M-HIRISE-5-DTM-V1.0
- HiRISE DTM COG (AWS): https://registry.opendata.aws/nasa-usgs-mars-hirise-dtms/ (bucket `s3://astrogeo-ard/mars/mro/hirise/controlled/dtm`)
- HiRISE images COG (AWS): https://registry.opendata.aws/nasa-usgs-mars-hirise
- CTX DTMs (AWS): https://registry.opendata.aws/nasa-usgs-controlled-mro-ctx-dtms
- USGS STAC docs: https://stac.astrogeology.usgs.gov/docs/data/
- MOLA + HRSC blended DEM (200 m): https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m ; file https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
- CTX global mosaic (5 m): https://murray-lab.caltech.edu/CTX/ ; USGS entry https://astrogeology.usgs.gov/search/map/mars-mro-ctx-global-mosaic-murray-lab-v1
- ESA PSA (HRSC, OMEGA): https://www.cosmos.esa.int/web/psa/mars-express
- Mars Trek WMTS: https://api.nasa.gov/mars-wmts/catalog
- Mars Orbital Data Explorer (root): https://ode.rsl.wustl.edu/mars/
- PDS Imaging (root): https://pds-imaging.jpl.nasa.gov/
- PDS Geosciences (root): https://pds-geosciences.wustl.edu

**Mineralogy, thermal, geology**
- CRISM: https://pds-geosciences.wustl.edu/missions/mro/crism.htm
- CRISM MTRDR: https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MRO-M-CRISM-5-RDR-MPTARGETED-V1.0
- THEMIS/TES thermal inertia: https://www.mars.asu.edu/data
- THEMIS thermal inertia (USGS, tiled): https://astrogeology.usgs.gov/search/map/themis_thermal_inertia_mosaic_quantitative_30s060e_100mpp
- THEMIS mosaics (AWS): https://registry.opendata.aws/nasa-usgs-themis-mosaics
- Jezero geologic map (Sun & Stack 2020, SIM 3464): https://pubs.usgs.gov/publication/sim3464 ; pamphlet https://pubs.usgs.gov/sim/3464/sim3464_pamphlet.pdf ; DOI https://doi.org/10.3133/sim3464

**Atmosphere, weather, radiation**
- Mars Climate Database: https://www-mars.lmd.jussieu.fr/ ; access https://www-mars.lmd.jussieu.fr/mars/access.html
- MCS: https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/mcs.html
- PDS Atmospheres Mars index: https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/data_archive.html
- MEDA: via https://pds-geosciences.wustl.edu/missions/mars2020/
- REMS: https://pdsatmospheres.nmsu.edu/data_and_services/atmospheres_data/MARS/curiosity/rems.html
- InSight: https://pds-geosciences.wustl.edu/missions/insight/index.htm
- MSL RAD: https://pds-ppi.igpp.ucla.edu/data/MSL-M-RAD-3-RDR-V1.0/AAREADME.TXT ; CDAWeb https://cdaweb.gsfc.nasa.gov/
- MARCI: via PDS Imaging (search MARCI)

**Resources and subsurface**
- SWIM: https://swim.psi.edu
- Odyssey GRS: https://pds-geosciences.wustl.edu/missions/odyssey/grsspecial.html
- SHARAD: https://pds-geosciences.wustl.edu/missions/mro/sharad.htm

**Geometry and ground truth**
- SPICE MRO: https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/mrosp_1000/aareadme.txt
- SPICE MAVEN: https://naif.jpl.nasa.gov/pub/naif/pds/pds4/maven/maven_spice/readme.txt
- SPICE Odyssey: https://pds.nasa.gov/data/ody-m-spice-6-v1.0/odsp_1000/aareadme.htm
- SPICE general (root): https://naif.jpl.nasa.gov/naif/data.html
- TGO / ESA kernels: ESA SPICE service (not verified)
- Perseverance traverse (PLACES): https://pds-geosciences.wustl.edu/missions/mars2020/places.htm
- Mars 2020 hub: https://pds-geosciences.wustl.edu/missions/mars2020/

**Rock-detector labels and tools**
- Public boulder outlines (Detectron2-ready): https://zenodo.org/records/8171052 ; raw/labeled https://zenodo.org/records/14250970
- RMaM-2020 rockfall set (pretraining only): https://edmond.mpdl.mpg.de/imeji/collection/DowTY91csU3jv9S2
- Golombek InSight rock data (abundance check only): https://data.caltech.edu/records/0ah85-w5237
- AI4Mars (rover-view; license not specified on the NASA portal): https://zenodo.org/records/15995036
- PyTorch install selector: https://pytorch.org/get-started/locally/
- Segment Anything: https://github.com/facebookresearch/segment-anything (not verified)
- Detectron2: https://github.com/facebookresearch/detectron2 (not verified)

## 6. Known pitfalls

- **Control mismatch:** controlled DTMs may not register with uncontrolled HiRISE images. Measure and budget the offset.
- **DTM artifacts:** box-shaped elevation steps and CCD-seam lines exist. Inspect hillshade; feed this into the DEM error model.
- **Licensing:** HRSC is CC BY-SA (share-alike on derived products). MCD is non-commercial. AI4Mars license is unspecified. Keep a licensing file.
- **Coarse layers:** THEMIS thermal inertia is ~100 m, GRS is hundreds of km, SWIM is ~3 km. Never present them as EVA-scale detail.
- **PLACES coverage:** Release 15 covers 2021-02-18 through 2026-02-03 (sols 0-1762) as of the check.
- **MCS:** no data for Sep-Nov 2009. MCD v6.2 was expected around mid-2026; check the current version.
- **No public Jezero-wide rock mask** was found. Labeling is a real cost.
- **Rover path bias:** it reflects operator choices and science goals, not feasibility limits.

## 7. Open items

1. Rules: pre-event work allowed? AI-use disclosure? (lead to confirm)
2. Final AOI from the coverage audit; second site and negative-case coverage.
3. Names for P1-P6 and machine assignments.
4. CesiumJS Mars-terrain workflow vs deck.gl/Three.js.
5. TGO/ESA SPICE kernel source.
6. Whether the Oct 28 challenge statement changed requirements or listed official datasets.
7. Current repo and work status.

## 8. Working agreement for the coding agent

- Claude writes code, tests and docs. Owners run on real data, report tracebacks and metrics, review and own their module.
- **Never fabricate** data values, citations, metrics or links. If something is unverified, say so in code comments and the README.
- The sandbox has no access to AWS, Zenodo, PDS or ODE and no GPU. Tests must run offline on synthetic fixtures. Code that touches the network or GPU is marked UNTESTED ON REAL DATA until an owner confirms it.
- Every module README states inputs, outputs, units, assumptions and validity range.
- Layers declare hard / objective / informational. Missing data is unknown.
- Conventions: Python 3.11, `src/` layout, pytest, ruff, type hints, deterministic seeds, pydantic schemas, small commits, CI on every push.
- Reports are terse: what passed, what is untested on real data, what is blocked.
- Ask only blocking questions; otherwise decide, document the decision, and continue.
- Do not build ML before the baseline planner passes its tests.
