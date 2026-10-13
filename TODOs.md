# Dhara Project TODOs & Innovation Roadmap
*Multimodal AI Hackathon 2026 — Track B: Flood & Debris Damage, Infrastructure Loss, and Isolation Intelligence*

This document tracks all features, architectural improvements, and innovations identified through competitive analysis across open hackathon repositories (`eo-rasuwa`, `baadhi`, `floodwatch-space`, `jalnetra`, `rasuwa-bulletin`), organizer clarifications, and geospatial disaster engineering principles.

---

## ⛔ Priority 0: Own Sentinel-1 processing chain from raw GRD (organizer requirement)
Organizers (Discord, 2026-10-13): Sentinel Hub data is already terrain-corrected, so "you still need
to make a processing pipeline for data, as the point is to process the raw stream coming from the
satellites." Today every S1 pass comes pre-processed from the Sentinel Hub Process API
(`src/dhara/adapters/sentinel1.py`: backCoeff, orthorectify on COP-DEM, LEE 5x5).
- [ ] Download original S1 IW GRD products from CDSE (allowlisted host; same-track pairs as now).
- [ ] Apply orbit file, thermal noise removal, radiometric calibration (β⁰/σ⁰ from the annotation LUTs).
- [ ] Speckle filter (Lee or refined Lee) and terrain flattening / terrain correction on Copernicus DEM
      GLO-30 (candidate: `sarsen`, Apache-2.0), output γ⁰ on the job's UTM grid with a layover/shadow mask.
- [ ] Make it the default; keep Sentinel Hub as a fast option (`--s1-source sentinelhub`).
- [ ] Compare both chains on the Trishuli case (pixel agreement, mapped km², buildings hit) for the report.
- [ ] Document disk and time cost per scene; cache the processed γ⁰ like the current passes.

---

## 🚀 Priority 1: Hydrological Wave Simulation & Kinematics Engine (Cool / Novel Feature)
- [x] **Dynamic Velocity Model along Copernicus DEM Flowpath:**
  - Implemented physics-based debris surge propagation speed $v(s)$ as a function of local channel slope $S(s)$ and hydraulic conveyance:
    $$v(s) = \min(v_{\max}, \max(v_{\min}, k \cdot \sqrt{S(s)}))$$
  - Calibrated against Trishuli flood observations (source ice-avalanche surge $\sim 190\text{ km/h}$ near Lhende, decelerating to $\sim 28\text{ km/h}$ at Betrawati and Galchhi).
- [x] **Downstream Evacuation Time Window:**
  - Integrated arrival time along the D8 steepest descent line:
    $$t_{\text{arrival}}(s) = \int_0^s \frac{d\xi}{v(\xi)}$$
  - Output actionable `evacuation_window` (`+Xm`), `wave_speed_kmh`, and `arrival_time_min` for every settlement along the downstream valley corridor.
- [x] **Height-Above-River ($\Delta z_{\text{river}}$) Settlement Pruning:**
  - Computed cross-sectional elevation delta between settlement coordinates and the nearest riverbed cell on Copernicus DEM.
  - Distinguishes *direct path* ($\Delta z \le 15\text{ m}$ or `hand_m` $\le 12\text{ m}$), *fringe* ($15 < \Delta z \le 30\text{ m}$), and *ridge safe* ($\Delta z > 30\text{ m}$) to avoid false positive damage flags on perched villages.
- [x] **MapLibre GL Kinematic Particle / Pulse Animation:**
  - Created dynamic flowpath surge wave pulse animation in `map-view.tsx` with animated line-dash phase cycling down the gorge and click-to-fly settlement markers with arrival timestamps.

---

## 🛰️ Priority 2: Multi-Sensor ML Ensemble & Statistical False-Alarm Control
- [x] **Placebo Null-Hypothesis Exceedance Layer (Statistical Calibration):**
  - Used pre-event same-track Sentinel-1 baseline pairs (`pair.pre2` vs `pair.pre`) to establish the empirical null distribution of backscatter change $\Delta \sigma^0$.
  - Detection threshold strictly bounds false-positive rate under the null to $p_{99}$ exceedance rate, eliminating false alarms caused by mountain radar shadow, foreshortening, and seasonal soil moisture shifts.
- [ ] **Deployment of 8-Model Stacking Ensemble in Inference Service:**
  - Package trained checkpoints from Kaggle dual T4 GPUs (`UNet++ ResNet-50`, `PAN ResNet-50`, `DeepLabV3+ ResNet-50`, `SegFormer mit_b0`, `Physics-Guided SAR U-Net`).
  - Compute pixel-level Epistemic Uncertainty ($\sigma_{\text{epistemic}} = \text{std}(p_1, \dots, p_K)$) alongside the ensemble consensus mask.
  - Export uncertainty GeoTIFF and expose confidence layers in the dashboard.
- [x] **Multi-Dataset Benchmark Showcase:**
  - Surfaced comparative evaluation matrix across Kuro Siwo test events and Himalayan zero-shot validation via `ModelBenchmarkDialog` in header.
  - Documents F1, IoU, false-alarm reduction, and mountain terrain degradation across baseline Otsu, UNet, SegFormer, DeepLabV3+, and Dhara Stacking Ensemble.

---

## 🗺️ Priority 3: Road Network Graph & Travel-Time Degradation
- [x] **Speed-Weighted Travel Time Graph:**
  - Upgraded OSM network edge weights from simple distance to travel duration based on highway hierarchy (`trunk`: 50 km/h, `secondary`: 30 km/h, `track`: 10 km/h, `path`: 4 km/h).
  - Computed pre-event vs. post-event hospital reachability delay (`delay_hours_target`, `hours_to_hospital_pre`, `hours_to_hospital_post`) in hours for every cut-off settlement.
- [x] **Interactive Bridge Restoration Ranking on Map:**
  - Added interactive click handlers on the Bridges & Repairs priority table: clicking any candidate flies directly to the bridge or isolated community on the map with active target pin highlighting.

---

## 🤖 Priority 4: Interactive Tactical AI Copilot (Bilingual EN/Nepali)
- [x] **UI Integration of Grounded Assistant:**
  - Wired `src/dhara/assistant/service.py` to `web/src/features/panels/results-panel.tsx`.
  - Added precision intent-specific answers for:
    - *"Which bridges are likely lost? / Top repair priority"*
    - *"Summary of isolated villages"*
    - *"Explain unobservable area"*
    - Multilingual support for English, Nepali, Hindi, Spanish, French.
- [x] **Official PDF / Print SitRep Export:**
  - Added 1-click printable SitRep export (`@media print` stylesheet) generating clean, professional, official NDRRMA-style field briefings without UI chrome.

---

## 📦 Priority 5: Zero-Config Demo Fixture & Live Judging Readiness
- [x] **Instant Offline Demo Fixture Mode:**
  - Implemented automatic initial load of the latest completed ground-truth run (`c8fb520f6174` / `bd970069311d`) so judges immediately see full damage maps and SitReps upon opening the app.
  - Added dedicated "Rasuwa 2026 Ground Truth" button in the top header for instant 1-click reset to the case study scenario.
- [x] **Before / After Comparison Blend Tool:**
  - Added floating comparison popover on the map canvas featuring a real-time 0-100% opacity slider and one-click "Pre-Event Baseline" vs "Post-Disaster Impact" toggle.

---

## 📄 Priority 6: Submission Deliverables (Report & Demo Video)
- [x] **6-Page Technical Report:**
  - Penned comprehensive academic technical report in `reports/TECHNICAL_REPORT.md` detailing mathematical formulation, multimodal remote sensing, Placebo null-hypothesis calibration, network percolation theory, and EMSR927 case study validation.
- [ ] **Bilingual Situation Report (PDF/HTML):**
  - Verify formatting and typography for English and Nepali official NDRRMA-style situation reports.
- [ ] **3-Minute Demo Video:**
  - Write a crisp recording script covering: (1) Area selection & date picking, (2) Sentinel-1/2 & DEM ingestion, (3) Flood & debris segmentation with epistemic uncertainty, (4) Flowpath wave kinematics simulation, (5) Network cut-off percolation & bridge ranking, (6) Grounded bilingual AI SitRep briefing.

---

## 🏫 Priority 7: Where, What, How — Named Schools, Facilities & Official Early Warnings
- [x] **Critical Social Infrastructure Tagging (Where & What):**
  - Parsed pre-event OSM tags for schools (`amenity=school/kindergarten/college`), health centers (`amenity=hospital/clinic/health_post`), and public buildings in `src/dhara/damage/overlay.py`.
  - Preserved tagged facilities in `buildings.geojson` regardless of hit threshold (`src/dhara/jobs/exports.py`), allowing responders to inspect every critical asset.
  - Added dedicated **Facilities Tab** in `web/src/features/panels/results-panel.tsx` with searchable schools/clinics, damage status, hazard exposure, and click-to-fly zoom.
- [x] **Official Early Warnings Retrieval & Banner:**
  - Ingested official DHM (Department of Hydrology & Meteorology) river station alerts (Trishuli at Betrawati, Bhotekoshi at Syabrubesi, Budhi Gandaki, Narayani at Devghat) and NDRRMA bulletins in `src/dhara/warnings/bulletins.py`.
  - Exposed `GET /api/warnings` endpoint and added interactive pulsing warning banner in `web/src/features/panels/results-panel.tsx`.
- [x] **Damage Detection Model Architecture (YOLO vs. Multi-Sensor Physics):**
  - Documented resolution limits: 10m Sentinel resolution spans only 1-2 pixels per rural house, where YOLO bounding boxes hallucinate.
  - Implemented the two-tier paradigm:
    1. *Core 10m EO Mode:* Pre-event OSM polygon intersection with SAR $\Delta \sigma^0$ + optical BSI/NDVI + DEM HAND (honest & compliant).
    2. *VHR / Drone Plug-in Mode:* Fine-tuned YOLOv8-OBB / SAM2 (Segment Anything) on high-resolution orthomosaics for instance-level structural roof grading.
