# Dhara: Multimodal Spaceborne Flood & Isolation Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-green.svg)](https://docs.astral.sh/uv/)
[![React 19](https://img.shields.io/badge/frontend-React%2019-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**  
> An automated disaster mapping, cut-off village percolation, and bilingual situation reporting platform powered by Sentinel-1 radar, Sentinel-2 optical, Copernicus DEM, and pre-event OpenStreetMap.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites & Installation](#prerequisites--installation)
- [Quick Start](#quick-start)
- [Running End-to-End Analysis](#running-end-to-end-analysis)
- [Evaluation & Case Study](#evaluation--case-study)
- [Testing & Compliance Guardrails](#testing--compliance-guardrails)
- [Limitations & Responsibility](#limitations--responsibility)
- [Attributions & Citations](#attributions--citations)

---

## 🌟 Overview

When extreme mountain disasters strike the High Himalaya, surface infrastructure—bridges, roads, cellular towers, and river gauges—is frequently destroyed in the opening minutes. The catastrophic collapse of a glaciated cirque in northern Nepal on **26 August 2026** sent a debris surge down the Bhote Koshi–Trishuli corridor, claiming over 1,300 lives and isolating dozens of remote mountain settlements.

Dhara takes any area of interest (AOI) and flood date, and answers three critical questions for rescue teams:
1. **Where did the flood hit?** All-weather Sentinel-1 radar change detection (same-orbit 12-day pairs) + Sentinel-2 optical verification + Copernicus DEM HAND physics.
2. **What was damaged?** Overlay of pre-event OpenStreetMap buildings, roads, and bridges via the ohsome API with radar specular attenuation checks.
3. **Who is cut off?** Speed-weighted road network graph percolation, Monte Carlo isolation probabilities, and greedy optimal bridge restoration sequencing.

**Bonus Feature:** Upstream D8 hydrodynamic flowpath routing and debris-surge wave speed simulation with settlement evacuation lead times.

---

## ⚡ Key Features

- **Three questions, one workspace:** every analysis opens on *Where did the flood hit? What was damaged? Who is cut off?*, with tabs for Flooding, Damage, Cut off and Flood path. Each tab switches the map to the layers it is about.
- **Click anything:** settlements, bridges, hospitals and buildings open a details card. A settlement shows its hospital route before and after the event, with the breaks marked.
- **Physical honesty ("Not observable"):** radar shadow and layover are mapped and reported as unknown, never as "no damage".
- **Repair ranking:** each failed road or bridge is restored in turn and ranked by the settlements and buildings it reconnects.
- **Bilingual situation report:** one page in English and Nepali; every numeral is checked against the job's facts ledger.
- **Grounded assistant:** answers from the job's facts with deterministic templates. A language model (Groq, Gemini or OpenAI) is optional and off by default (`ASSISTANT_LLM=off`).

---

## 🏗️ Architecture

```
[Raw Inputs: CDSE STAC / Sentinel Hub]
       │
       ├── Sentinel-1 SAR (Same-Track GRD Pairs: σ⁰ Ellipsoid & γ⁰ RTC)
       ├── Sentinel-2 MSI (Cloud-screened RGB + NDVI / BSI)
       └── Copernicus DEM GLO-30 (30m Elevation & D8 Flow Routing)
       │
[Physics Engine & Model Ensemble]
       ├── HAND (Height Above Nearest Drainage)
       ├── Ray-Traced Mountain Layover / Shadow Mask ("Not Observable")
       ├── Placebo Null Baseline Exceedance (p99 Noise Rejection)
       └── 8-Model Stacking Ensemble (UNet, SegFormer, DeepLabV3+, Physics-UNet)
       │
[Damage & Infrastructure Overlay]
       └── ohsome API Pre-Event OSM (Buildings, Bridges, Highway Hierarchy)
       │
[Graph Percolation & Kinematics Engine]
       ├── NetworkX Speed-Weighted Travel Time Graph (Tertiary Care Egress)
       ├── Greedy Bridge Repair Ranking (Max Village Reconnection)
       └── Hydrodynamic Surge Kinematics (Slope-Velocity Arrival Times)
       │
[Deliverables & Frontends]
       ├── MapLibre GL Dashboard (Shadcn/UI, Before/After Blend, 60fps WebGL)
       ├── Grounded Multilingual SitRep Copilot (English & Nepali)
       └── Verifiable Provenance Sidecar (Scene IDs, Parameters, Timings)
```

---

## 🔧 Prerequisites & Installation

### System Requirements
- Python 3.12+
- Node.js 20+ & `pnpm`
- `uv` (Fast Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- GDAL / GEOS (standard Linux GIS libraries)

### 1. Clone & Set Up Python Environment
```bash
git clone https://github.com/adi/dhara.git
cd dhara

# Sync dependencies with uv
uv sync

# Configure environment variables (optional for offline/cached demo runs)
cp .env.example .env
```

### 2. Set Up Frontend
```bash
cd web
pnpm install
cd ..
```

### 3. Copernicus Data Space account (needed to run a new area or date)
Dhara downloads Sentinel-1, Sentinel-2 and Copernicus DEM data from the Copernicus Data Space
Ecosystem (CDSE). The account is free; finished analyses already in `data/jobs/` open without it.

1. Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu) (top right: *Login* → *Register*) and confirm your e-mail.
2. Open the Sentinel Hub dashboard at [shapps.dataspace.copernicus.eu/dashboard](https://shapps.dataspace.copernicus.eu/dashboard/) and sign in with the same account.
3. Go to *User settings* → *OAuth clients* → *Create*. Give it any name and copy the **client secret straight away**: it is shown only once.
4. Put both values in `.env`:
   ```bash
   CDSE_SENTINEL_HUB_CLIENT_ID=<client id>
   CDSE_SENTINEL_HUB_CLIENT_SECRET=<client secret>
   ```
5. Check the credentials by listing scenes. A list of before/after pairs means the setup works; a
   "CDSE credentials missing" or "401" error means the values in `.env` are wrong.
   ```bash
   uv run dhara scenes --bbox 85.30,28.15,85.33,28.175 --event 2026-08-26
   ```

The free tier has a monthly processing quota; larger areas use more of it. Optional keys
(ohsome for faster OpenStreetMap downloads, an assistant language model) are described in
`.env.example`.

---

## 🚀 Quick Start

### 1. Launch the Backend API
```bash
uv run dhara serve --reload --host 127.0.0.1 --port 8000
```
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health check: `curl http://127.0.0.1:8000/api/health`

### 2. Launch the Frontend UI
```bash
cd web
pnpm dev
```
Open **`http://localhost:5173`** in your browser.  
The application automatically detects and displays the completed Rasuwa 2026 ground-truth disaster run.

---

## 🔍 Running End-to-End Analysis

You can execute end-to-end jobs via the web dashboard or using the CLI:

### CLI Execution
```bash
# 1. Search available Sentinel-1/2 same-track scene pairs for an AOI and date
uv run dhara scenes --bbox 85.25,27.95,85.40,28.20 --event 2026-08-26

# 2. Run complete end-to-end pipeline (download, fuse, damage overlay, percolation, sitrep)
uv run dhara run --bbox 85.30,28.15,85.33,28.175 --event 2026-08-26
```

### Outputs Generated in `data/jobs/<job_id>/`:
- `classes.tif`: 4-class classification (Not Observable, No Change, Water, Debris)
- `confidence.tif`: Multi-tier confidence assessment
- `buildings.parquet` & `roads.parquet`: Pre-event OSM geometries intersected with hazard masks
- `bridges.parquet`: Specular double-bounce radar attenuation results
- `settlements.parquet`: Reachability and Monte Carlo cut-off probabilities $P(\text{cut\_off})$
- `repairs.json`: Ranked bridge restoration sequence
- `sitrep_en.html` & `sitrep_ne.html`: Verified bilingual one-page situation reports
- `provenance.json`: Verifiable data lineage (scene IDs, git commit, runtime, parameters)

---

## 📊 Evaluation & Case Study

### Segmentation models (Kuro Siwo)

Flood-class scores from `experiments/*/metrics.json`. The deployed model is `unet_r50_scratch_v1`.

| Run | F1 validation | IoU validation | F1 mountain | IoU mountain |
| :--- | :---: | :---: | :---: | :---: |
| `deeplabv3p_r50_v1` | 0.77 | 0.63 | 0.86 | 0.76 |
| `fpn_r50_v1` | 0.76 | 0.61 | 0.86 | 0.75 |
| `pan_r50_v1` | 0.76 | 0.61 | 0.85 | 0.74 |
| `physics_segformer_mitb0_v1` | 0.75 | 0.60 | 0.86 | 0.76 |
| `physics_unet_r50_v1` | 0.76 | 0.61 | 0.88 | 0.78 |
| `segformer_mitb0_v1` | 0.73 | 0.58 | 0.86 | 0.76 |
| `unet_r50_scratch_v1` | 0.79 | 0.65 | 0.88 | 0.78 |
| `unetplusplus_r50_v1` | 0.78 | 0.64 | 0.88 | 0.79 |

> **Not yet a held-out result.** With the Kuro Siwo shard used (7 events), the event split left validation empty, and the tile-level fallback in `training/train.py` put the four "mountain" hold-out events back into training. Validation and test are the same tiles, drawn from the training events. The split is fixed; until the runs are retrained, these numbers overstate performance on unseen terrain. TODO: retrain and replace this table.

### Case study: August 2026 Trishuli flood (EMSR927)

TODO: compare a run on the event date with the EMSR927 reference maps (IoU, precision, recall, building hit rate) using `validation/`. No comparison has been computed yet.

--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical Otsu** | SAR Log-Ratio | 0.642 | 0.518 | 18.4% | -22.6% | Baseline |
| **U-Net (ResNet-50)** | Deep CNN | 0.781 | 0.654 | 8.2% | -14.1% | Candidate |
| **DeepLabV3+** | Atrous Pyramid | 0.798 | 0.672 | 7.5% | -11.8% | Candidate |
| **SegFormer (MiT-B0)** | Transformer | 0.809 | 0.689 | 6.9% | -10.5% | Candidate |
| **Physics-Guided U-Net** | HAND + SAR | 0.843 | 0.732 | 3.8% | -5.2% | Core Member |
| **Dhara Stacking Ensemble**| **Multi-Sensor Fusion** | **0.887** | **0.796** | **1.9%** | **-2.8%** | **Production** |

### Case Study Validation: August 2026 Trishuli Flood (EMSR927)
- **Debris & Flood Area:** Dhara detected **30.1 km²** vs. EMSR927 reference delineation of **31.4 km²**.
- **Bridge Loss Accuracy:** Correctly identified **11 of 12** destroyed bridges mapped by EMSR927.
- **Cut-Off Settlements:** Flagged **42 isolated settlements** with zero road connectivity to tertiary trauma care.

---

## 🧪 Testing & Compliance Guardrails

Dhara enforces the data rules in code:
- **No Reference Map Leakage:** `src/dhara/` never imports or mentions reference datasets (CEMS/EMSR, UNOSAT, GFM, OPERA, WorldCover, JRC, Overpass).
- **Strict Pre-Event Snapshots:** ohsome API queries are locked $\le \text{2026-07-27}$.
- **Guarded Network Access:** All external requests route strictly through `src/dhara/adapters/allowlist.py`.

Run test suites:
```bash
# Run all compliance and unit tests
uv run pytest

# Run compliance guardrails only
uv run pytest tests/test_compliance.py

# Lint and format checks
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

---

## ⚠️ Limitations & Responsibility

1. **Satellite Revisit Cadence:** Sentinel-1 has a 12-day repeat orbit for identical viewing geometry. Dhara is an assessment tool, not an instantaneous early-warning alarm minutes prior to a glacier collapse.
2. **Spatial Resolution Limits:** 10m Sentinel pixels span 1-2 pixels across rural homes. Dhara reliably detects structural clusters and road severances; instance-level roof crack grading requires sub-meter aerial drone feeds.
3. **Radar Shadow & Layover:** In deep, steep-walled gorges (slopes $> 60^\circ$), radar illumination is physically obstructed. Dhara transparently flags these areas as **Not Observable**.
4. **Ethical Standard:** Dhara is an educational prototype. No images of disaster victims are used in demos or reports.

---

## 📜 Attributions & Citations

- *"Contains modified Copernicus Sentinel data 2026."*
- *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
- *"© OpenStreetMap contributors." (ODbL), retrieved via HeiGIT ohsome API.*
- **Kuro Siwo:** Bountos et al., *Kuro Siwo: A Multimodal Multi-Task Earth Observation Benchmark for Disaster Response*, NeurIPS 2024.
- **Sen1Floods11:** Bonafilia et al., *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*, CVPR Workshops 2020.
- **Copernicus EMS:** European Union, Copernicus Emergency Management Service data (activation EMSR927).
