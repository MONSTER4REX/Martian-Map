# Data licenses and provenance

The code in this repo is MIT-licensed. **Input datasets are not redistributed
here** and carry their own terms. This file tracks the license and any
share-alike / non-commercial obligations that flow into derived products.
Owners: fill the "Verified by / date" column when you confirm a license at the
source — do not take the notes below as authoritative.

> Status legend: **UNVERIFIED** = copied from the planning brief, not yet
> confirmed at source. Confirm before shipping any derived product.

| Dataset | Source | License (claimed) | Obligations on derived products | Status |
|---|---|---|---|---|
| HiRISE DTMs / images (COG on AWS) | USGS Astrogeology / NASA PDS | NASA data, generally public domain | Cite mission + producer | UNVERIFIED |
| CTX DTMs / global mosaic (Murray Lab) | Caltech Murray Lab / USGS | Check Murray Lab terms | Cite; check attribution terms | UNVERIFIED |
| MOLA + HRSC blended DEM 200 m | USGS / ESA | **HRSC is CC BY-SA** | **Share-alike** on derived products that include HRSC | UNVERIFIED |
| THEMIS thermal inertia | ASU / USGS | NASA data | Cite; ~100 m — never present as EVA-scale | UNVERIFIED |
| CRISM MTRDR | NASA PDS | NASA data | Cite | UNVERIFIED |
| Jezero geologic map (SIM 3464) | USGS | USGS publication | Cite Sun & Stack 2020 | UNVERIFIED |
| Mars Climate Database | LMD | **Non-commercial** | No commercial use of MCD-derived products | UNVERIFIED |
| MEDA / REMS / RAD | NASA PDS | NASA data | Cite | UNVERIFIED |
| PLACES (Perseverance traverse) | USGS / NASA | NASA data | Cite release number | UNVERIFIED |
| SPICE kernels (MRO/MAVEN/Odyssey/TGO) | NAIF / ESA | Public; ESA kernels separate terms | Cite kernel set + version | UNVERIFIED |
| Boulder outlines (Zenodo 8171052 / 14250970) | Zenodo | Per-record (check CC variant) | Per record | UNVERIFIED |
| RMaM-2020 rockfall set | MPDL Edmond | Pretraining only per brief | Do not ship as labels | UNVERIFIED |
| AI4Mars | Zenodo / NASA | **License unspecified on NASA portal** | Do not use until clarified | UNVERIFIED |

## Flow-down rules (apply when they are confirmed)

- **CC BY-SA (HRSC):** any shipped raster/product that mixes in HRSC must be
  released share-alike and must attribute. Keep HRSC-derived layers separable so
  non-SA products can exclude them.
- **MCD non-commercial:** mark MCD-derived layers non-commercial; exclude them
  from any commercial demo.
- **Unspecified (AI4Mars):** treat as "do not redistribute / do not train on"
  until a license is confirmed in writing.

All dataset URLs live in `MARSWALK_CONTEXT.md` §5; rows there marked *root* and
items marked "not verified" still need a first-hand check.
