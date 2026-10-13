# Dhara: Multimodal Satellite Flood Damage, Infrastructure Loss, and Isolation Intelligence
**Technical Report — Multimodal AI Hackathon 2026 (Track B: Remote Sensing & Disaster Response)**
*Author / Team: Project Dhara*  
*Date: October 2026*  

---

## 1. Executive Summary & Problem Formulation

When extreme mountain disasters strike the High Himalaya, surface infrastructure—bridges, roads, cellular towers, and river gauges—is frequently destroyed in the opening minutes. The catastrophic collapse of a glaciated cirque in northern Nepal on **26 August 2026** sent a debris surge of ice, pulverized granite, and mud down the Bhote Koshi–Trishuli corridor, claiming over 1,300 lives and isolating dozens of remote mountain settlements.

In such high-relief catastrophes, ground truth is absent when decisions matter most. Optical satellite observations are hampered by persistent monsoon cloud cover, while steep mountain gorges induce severe geometric radar layover and radar shadow in synthetic aperture radar (SAR). Furthermore, simple flood intersection fails operationally: a village whose houses are physically dry may nonetheless experience catastrophic isolation if its sole access bridge is swept away.

**Dhara** is an end-to-end multimodal disaster intelligence platform engineered to answer three life-critical questions for emergency rescue coordinators from space:
1. **Where did the flood and debris hit?** (Multimodal Sentinel-1 SAR change detection + Sentinel-2 optical verification + Copernicus DEM HAND physics).
2. **What was damaged?** (Pre-event OpenStreetMap building, road, and bridge intersection with radar specular backscatter attenuation checks).
3. **Who is cut off?** (Dynamic network graph percolation, hospital egress travel-time degradation, and greedy bridge restoration sequencing).

---

## 2. Multimodal Remote Sensing Pipeline

Dhara enforces strict operational independence and zero-disqualification compliance: it relies exclusively on raw, openly accessible Copernicus data, Copernicus DEM GLO-30, and pre-event OpenStreetMap snapshots via the ohsome API ($\le \text{2026-07-27}$).

```
[CDSE STAC / Sentinel Hub]
       │
       ├── Sentinel-1 SAR (Same-Track GRD Pairs: σ⁰ Ellipsoid & γ⁰ RTC)
       ├── Sentinel-2 MSI (Cloud-screened RGB + NDVI / BSI)
       └── Copernicus DEM GLO-30 (30m Elevation & D8 Flow Routing)
       │
[Data Fusion & Physics Engine]
       ├── HAND (Height Above Nearest Drainage)
       ├── Ray-Traced Mountain Layover / Shadow Mask ("Not Observable")
       ├── Placebo Null Baseline Exceedance (p99 Noise Rejection)
       └── 8-Model Stacking Ensemble (UNet, SegFormer, DeepLabV3+, Physics-UNet)
       │
[Damage & Infrastructure Overlay]
       └── ohsome Pre-Event OSM (Buildings, Bridges, Highway Hierarchy)
       │
[Graph Percolation & Kinematics]
       ├── NetworkX Speed-Weighted Travel Time Graph (Tertiary Care Egress)
       ├── Greedy Bridge Repair Ranking (Max Village Reconnection)
       └── Hydrodynamic Surge Kinematics (Slope-Velocity Arrival Times)
       │
[Deliverables & Frontends]
       ├── MapLibre GL Dashboard (Shadcn/UI, Before/After Blend, 60fps Pulse)
       ├── Grounded Multilingual SitRep Copilot (English, Nepali, Hindi)
       └── Verifiable Provenance Sidecar (Scene IDs, Parameters, Timings)
```

---

## 3. Multi-Sensor Physics & Machine Learning Architecture

### 3.1 SAR Radiometric Preprocessing
Standard dual-pol (VV/VH) Sentinel-1 Ground Range Detected (GRD) scenes are obtained from identical relative orbit tracks (12-day repeat cycle) to guarantee identical look geometry. Radiometric calibration is bifurcated based on the downstream member:
- **ML Member:** Linear $\sigma^0$ ellipsoid with Lee speckle filtering, matching the exact SNAP preprocessing pipeline of the Kuro Siwo training corpus.
- **Debris & Classical Member:** Radiometric Terrain Corrected (RTC) $\gamma^0$, normalizing for local incidence angle variations across $45^\circ+$ Himalayan valley walls.

### 3.2 Placebo Null-Hypothesis Exceedance Calibration
To eliminate false alarms caused by steep terrain geometry, soil moisture shifts, and speckle noise, Dhara computes an empirical null distribution of backscatter change $\Delta \sigma^0$ using pre-event baseline pairs on the same track ($\text{pair.pre2}$ vs $\text{pair.pre1}$). A post-event pixel is flagged only if its backscatter drop or rise exceeds the empirical 99th percentile ($p_{99}$) of the null distribution.

### 3.3 The "Not Observable" Class (Physical Honesty)
Unlike competing models that classify obscured pixels as "undamaged", Dhara computes exact layover and radar shadow masks combined with optical cloud probabilities. Obscured terrain is classed **"Not Observable"** ($A_{\text{unobs}} \approx 0.62\text{ km}^2$ in the Rasuwa gorge). This alerts mission planners that physical UAV reconnaissance is required before declaring an area safe.

### 3.4 Deep Learning Architecture Comparison & Ablation
Trained across Kuro Siwo and Sen1Floods11 corpora on dual NVIDIA T4 GPUs:

| Model Architecture | Type | F1 Score | IoU | False Alarm Rate | Mountain Drop | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical Otsu** | SAR Log-Ratio | 0.642 | 0.518 | 18.4% | -22.6% | Baseline |
| **U-Net (ResNet-50)** | Deep CNN | 0.781 | 0.654 | 8.2% | -14.1% | Candidate |
| **DeepLabV3+** | Atrous Pyramid | 0.798 | 0.672 | 7.5% | -11.8% | Candidate |
| **SegFormer (MiT-B0)** | Hierarchical Transformer | 0.809 | 0.689 | 6.9% | -10.5% | Candidate |
| **Physics-Guided U-Net** | HAND + SAR | 0.843 | 0.732 | 3.8% | -5.2% | Core Member |
| **Dhara Stacking Ensemble**| **Multi-Sensor Fusion** | **0.887** | **0.796** | **1.9%** | **-2.8%** | **Production** |

---

## 4. Road Network Percolation & Bridge Restoration Priority

### 4.1 Speed-Weighted Reachability Graph
Road network edges extracted from pre-event OSM are converted into a directed graph $G = (V, E)$ weighted by traversal duration $\tau(e) = L(e) / v_{\text{class}}$ based on highway functional hierarchy (`trunk`: 50 km/h, `secondary`: 30 km/h, `track`: 10 km/h, `path`: 4 km/h). Edges intersecting hazard masks are severed.

### 4.2 Monte Carlo Isolation Probability $P(\text{cut\_off})$
To account for hazard detection uncertainty, 500 Monte Carlo draws sample edge failures with probability proportional to hazard intersection confidence. Every settlement receives a calibrated isolation probability $P(\text{cut\_off})$ and delay penalty $\Delta t_{\text{hospital}}$.

### 4.3 Greedy Optimal Bridge Repair Algorithm
When dozens of bridges are severed, emergency engineering teams must deploy temporary Bailey bridges to maximize immediate humanitarian reach. Dhara iteratively evaluates the marginal reconnection yield of each damaged bridge:
$$\text{Yield}(b) = \Delta |V_{\text{reconnected settlement}}| + \lambda \cdot \Delta |B_{\text{reconnected buildings}}|$$
In the Rasuwa case study, fixing primary bridge `way/344083436` reconnects 3 cut-off communities and 380 structures, while repairing footway bridge `way/784422172` restores access to 7 remote villages (Rimche, Sherpagaun, Khangjim).

---

## 5. Hydrodynamic Debris-Surge Kinematics Engine

For any user-selected upstream avalanche or GLOF ignition point, Dhara routes a hydraulic flowpath along the D8 steepest descent corridor derived from Copernicus DEM:
$$v(s) = \min\left(v_{\max}, \max\left(v_{\min}, k \cdot \sqrt{S(s)}\right)\right)$$
Downstream settlements receive calculated evacuation lead times:
$$t_{\text{arrival}}(s) = \int_0^s \frac{d\xi}{v(\xi)}$$
Furthermore, settlements are categorized by elevation delta above the riverbed ($\Delta z_{\text{river}}$):
- **Direct Impact Zone:** $\Delta z \le 15\text{ m}$ (Immediate inundation and scour).
- **Fringe Zone:** $15\text{ m} < \Delta z \le 30\text{ m}$ (Erosion and foundation undercut).
- **Ridge Safe:** $\Delta z > 30\text{ m}$ (Perched villages safe from hydraulic inundation, preserving emergency helipads).

---

## 6. Rasuwa 2026 Ground-Truth Validation (EMSR927)

Evaluating Dhara on the 26 August 2026 Himalayan disaster and comparing against official Copernicus Emergency Management Service activation **EMSR927**:
- **Debris & Flood Area:** Dhara detected **30.1 km²** of active scour and debris, closely matching the EMSR927 delineation of 31.4 km².
- **Bridge Loss Accuracy:** 11 of 12 bridges marked "Destroyed" in EMSR927 were correctly flagged by Dhara's specular double-bounce attenuation test ($\Delta \sigma^0 \le -3.5\text{ dB}$).
- **Cut-Off Detection:** Dhara identified 42 isolated settlements with 0 road connectivity to district trauma care, validating NDRRMA field reports.

---

## 7. System Limitations & Honesty Notes

1. **Revisit Time Cadence:** Sentinel-1 has a 12-day repeat orbit for identical geometry. It cannot provide real-time warning minutes prior to glacier collapse.
2. **Spatial Resolution Bounds:** 10m Sentinel pixels span 1-2 pixels per rural Nepali home. Dhara measures multi-pixel structural clusters and road links; instance-level roof crack grading requires VHR sub-meter drone feeds.
3. **Severe Radar Shadow:** In north-facing gorges with slope $> 60^\circ$, radar illumination is impossible. Dhara transparently flags these areas as *Not Observable*.

---

## 8. Required Attributions & Citations

- *"Contains modified Copernicus Sentinel data 2026."*
- *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
- *"© OpenStreetMap contributors."*
- *Kuro Siwo Citation:* Bountos et al., NeurIPS 2024.
- *Sen1Floods11 Citation:* Bonafilia et al., CVPR Workshops 2020.
