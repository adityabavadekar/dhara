# Project ARGUS-Himal: A Design Blueprint for an Evidence-Graded, Network-Aware Flood/Debris-Flow Damage and Isolation Intelligence System (Track B, Rasuwa 2026)

The highest-value, most defensible "new" thing you can build is not a better flood segmenter. It is a probabilistic isolation engine. It turns pixel-level SAR/optical evidence of debris-flood impact into bridge- and road-segment failure probabilities. It then propagates those through the pre-event OpenStreetMap network to give every settlement a calibrated "probability of being cut off from the nearest hospital/district HQ", plus a ranked "who to reach first, and where a helicopter can land" list. The rest of the pipeline exists to feed that engine honestly: an ensemble of a Kuro Siwo-trained multi-temporal SAR model, a Sen1Floods11-fine-tuned S1+S2 foundation model, and GFM-style physics/threshold algorithms. This is exactly what failed on the ground in August 2026. The Department of Roads counted 41 motorable bridges destroyed and four more damaged, "leaving only 18 of the 63 bridges along the corridor intact" (Kathmandu Post, 8 Sep 2026). The Ministry of Physical Infrastructure reported "approximately 40 kilometers of road in Rasuwa" completely damaged. At least one village (Tiru) waited 15 days for a helicopter.\[1\]\[2\] A Timure helipad was itself swept away.\[3\] And no operational service, Copernicus EMS included, produced a network-level cut-off product.

## TL;DR

- **Build the core as "evidence → edge-failure probability → network reachability".** Use a three-member ensemble: UNet-ResNet50 on Kuro Siwo (best BlackBench GRD model, F1-flood 80.12%, mIoU 76.20%); TerraMind/Prithvi-EO-2.0 fine-tuned on Sen1Floods11 (TerraMind S1+S2 mIoU 90.62; Prithvi-EO-2.0-600M-TL water IoU 83.2); and a GFM-style log-ratio/split-Otsu/HAND classical member. Add an explicit debris/scour class from bidirectional SAR change plus coherence loss. Then compute Monte-Carlo settlement isolation probabilities on the ≤27 July 2026 OSM graph. This targets the 30% + 20% judging blocks with one coherent idea.
- **Choose AI option (a), segmentation with demonstrated Himalayan generalization, as the scored component.** Still ship a deterministic, numbers-from-the-database bilingual sitrep generator, where the LLM may only verbalize verified tokens. Debris flows do not look like floodwater in SAR (they are often bright/rough, not dark/smooth). Water-only models trained on Kuro Siwo/Sen1Floods11 will therefore under-detect the Rasuwa damage unless you add change-based debris evidence. Say this openly; it is your honesty-score win.
- **Keep a hard wall between the competition-compliant core and the plug-in layer.** The core uses only S1, S2, Copernicus DEM, OSM ≤27 July 2026, Kuro Siwo, and Sen1Floods11. The plug-in layer adds NISAR, EOS-04, OPERA, GPM IMERG, gauges/IoT, seismic triggers, news/NLP, and BRIGHT/xBD-trained damage models. The single biggest gray area is pretrained foundation-model weights (Prithvi/TerraMind pretrained on other data); get written organizer clarification or keep a weights-free fallback. Never ingest GFM/OPERA DSWx/UNOSAT/CEMS/HOT-fAIr outputs except in a sealed validation notebook.

## Key Findings

### 1. What actually happened, and what that implies for system design

- **Trigger and speed.** At about 08:40 NPT on 26 August 2026, a large section of glacier and adjacent bedrock failed above the Lhende river. USGS later said the seismic signal (registering as a magnitude 5.2 event) came from the landslide itself, not an earthquake.\[4\] Jeffrey Kargel (Planetary Science Institute) estimated the flood wave "descended the first 22 kilometers (13.7 miles) at an average speed of 193 kilometers per hour".\[5\] According to the ArcGIS StoryMap "August 2026 Nepal Trishuli Flood", "At Galchhi… roughly 88 km downstream from the source, the Trishuli reportedly rose by approximately nine meters (28 ft) in only 30 minutes".
- **Location is contested, and this matters for cross-border data.** The problem statement and some reports place the source in northern Nepal (Langtang Lirung north face, Langtang National Park).\[6\]\[7\] Al Jazeera described the collapse as "high above the Lhende river in Tibet", about 20 km northeast of the Rasuwagadhi crossing.\[4\] ICIMOD calls it an "ice-rock avalanche"; NDRRMA initially called it a "glacial-related flood"; USGS calls it a glacier collapse or mass movement.\[8\] Your system should be agnostic to the border: process the full upstream catchment from Sentinel/DEM data regardless of jurisdiction.
- **Ground sensing died first.** The flood destroyed four automatic hydrological stations (Syabrubesi, Rasuwa–Bhotekoshi, Betrawati, Malekhu/Phurke).\[9\]\[10\] DHM learned of the flood at 09:00 from the District Administrative Officer, "by which time several communications towers had been destroyed".\[7\] The flood reached the Narayani at Devghat around 15:20, peaking at 6.57 m at 16:00.\[9\]\[10\] That is roughly 6.5–7 hours of downstream travel time: the window a proactive system could exploit.
- **Damage is network damage.** The Ministry of Physical Infrastructure (Desh Sanchar, 27 Aug 2026) reported "approximately 40 kilometers of road in Rasuwa have been completely damaged… about 10 kilometers of the road from Galchi to Trishuli is also closed". The Department of Roads counted 41 motorable bridges destroyed and "Four others were damaged, leaving only 18 of the 63 bridges along the corridor intact" (Kathmandu Post, 8 Sep 2026); NDRRMA SitRep #1 (1 Sep) gives the same "41 bridges have been washed away and 4 bridges damaged". The cost estimates changed over time: on 29 Aug the Kathmandu Post reported Rs20 billion for roads and bridges, and by 8 and 13 Sep the Department of Roads put it at "around Rs30 billion in damage to roads and bridges alone", with rebuilding at "about Rs50 billion". The government sought 42 Bailey bridges from India and China. CARE's early count was 32 bridges, an example of how numbers evolve, which your sitrep must timestamp.\[11\] A school principal in Trishuli said "a helicopter is needed to travel between wards of the same municipality".\[12\] Isolation reached the trail level: Hakubeshi was cut off when a Bailey bridge and a suspension bridge to Dhunche were lost, and the Rasuwa District Administration Office (Asst. CDO Dhruba Prasad Adhikari, Radio Nepal, 5 Sep 2026) said the flood swept away "41 kilometers of road, six concrete bridges, eight Bailey bridges, and 10 suspension bridges in the district". Langtang Valley was cut off from Dhunche although not directly flooded.\[13\] Tiru received its first helicopter 15 days after the deluge.\[1\] At Timure, a rescue helicopter returned without landing because the local helipad had been swept away.\[3\]
- **Exposure.** One analysis storymap estimated 4,689 buildings in the flood-affected zone.\[14\] Copernicus imagery analysis reported more than 240 buildings destroyed and 32 damaged within the Syapru Besi area alone.\[15\] UN OCHA SitRep No. 7 (8 Sep 2026) reported "3,629 displaced people are living in 37 displacement sites in Nuwakot, Rasuwa, Dhading, and Kathmandu"; SitRep No. 6 (3 Sep) cited IOM DTM: "4,288 households were affected… of which 2,886 households were displaced". The Rasuwa DAO (Radio Nepal, 5 Sep 2026) listed eight hydropower projects as "completely damaged": 111 MW Rasuwagadhi, 120 MW Bhotekoshi, 20 MW Chilime, 20 MW Langtang Khola, 216 MW Upper Trishuli-1, 5 MW Mailung Khola, 60 MW Upper Trishuli-3A and 37.3 MW Trishuli 3B.

**Design implications.** (1) Cut-off analysis must include footpaths and suspension bridges, not just motorable roads, because isolation in Langtang/Tiru was trail-level. (2) Helicopter landing-zone (HLZ) candidates must be computed per isolated settlement, because pre-existing helipads failed. (3) The system must be useful within hours, before any VHR tasking arrives.

### 2. How the official response unfolded, and its gaps

- **Activations were fast.** The International Charter was activated on 26 Aug (Activation 1052, at 14:58 UTC+02:00, requested by ADRC on behalf of DHM, project-managed by UNITAR).\[10\]\[16\] Sentinel Asia was also activated.\[17\]\[18\] Copernicus EMS Rapid Mapping activated EMSR927 on 26 Aug "to provide flood extent and damage assessment emergency mapping".\[19\]\[20\] HOT/NAXA launched a Tasking Manager campaign on 27 Aug and shared a fAIr damage assessment on HDX on 28 Aug.\[21\]
- **EMSR927 product structure.** EU Space reported on 28 Aug that CEMS was mapping "across four Areas of Interest";\[15\] the activation later grew to six.\[18\] A third-party bot tracking the Copernicus API (not an official source) lists only Grading (GRA) products. These are AOI01 Syapru Besi (WorldView-3), AOI02 Timure (Legion, v2), AOI03 Bidur (PeruSAT-1, Resurs-P; Satellogic, BlackSky), and AOI05 Phosretar (Legion, Pléiades Neo; up to v3).\[22\]\[23\] AOI04 Bharatpur and AOI06 Kyundi were marked "not feasible", and the bot reported the activation closed by 14 Sep.\[24\] No First Estimate or Delineation products appear in that tracker; confirm against the official activation page before claiming this in your report.
- **Clouds and latency.** Nepali NGO NESRA FloodWatch noted early that "EMSR927 Rapid Mapping remains ongoing with two AOIs and no products displayed at the review time". It added that "Nine post-event optical scenes are public, but severe cloud limits coverage", and that PlanetScope and SkySat collects were "cloud-limited".\[18\] By 3 Sep, NESRA counted six AOIs and eight product records for EMSR927, and 21 products under Charter activation 1052.\[18\]
- **The gap you fill.** Every official product was an extent or building-grading map over limited AOIs, mostly from commercial VHR optical. None delivered: (a) a corridor-wide, cloud-independent first-pass map from Sentinel-1 within hours; (b) a settlement-level cut-off/reachability product; (c) bridge-specific loss inference; (d) HLZ suggestions; or (e) quantified uncertainty and provenance per claim. A clear Sentinel-2 pair did exist (12 Aug and 27 Aug over Betrawati/Gerkhu, showing "a much wider, sediment-filled river channel").\[14\]\[19\] So an S1-first, S2-when-clear design is validated by the event itself.

### 3. Model landscape: what the benchmarks actually say

Three facts should drive model choice:

1. **On Kuro Siwo (SAR, multi-temporal, the recommended set), plain CNN segmentation beats change-detection nets and ViTs.** UNet-ResNet50 with 3 captures reaches F1-flood 80.12% and mIoU 76.20%. FloodViT reaches 78.70/74.22, ChangeFormer 76.96/73.23, and SNUNet-CD 74.26/71.30. The authors note FloodViT "surpasses all change detection methods". DEM/slope inputs gave only marginal gains in BlackBench.\[25\] The newer TerraFlow (arXiv 2603.12762) with temporal attention reports beating the UNet-ResNet50 baseline in mIoU and F1-flood; its authors also describe Kuro Siwo as "close to saturation".\[26\]
2. **On Sen1Floods11, foundation models lead but margins are small, and geographic shift costs more than the model gap.** TerraMind reports Sen1Floods11 mIoU of 80.39 (S1 only), 89.57 (S2 only) and 90.62 (S1+S2).\[27\] Prithvi-EO-2.0-600M-TL reports mIoU 90.5 and water IoU 83.2.\[28\] Prithvi-CAFE (arXiv 2601.02315) reports mIoU 90.50 and water IoU 83.41 on test, and mIoU 88.87 on the Bolivia hold-out.\[29\] The PEFT study (arXiv 2504.17397) found mIoU drops by 7.31 pp on average on geographically held-out Sen1Floods11 sites. Even a randomly initialized UNet scored 90.75 test / 87.81 held-out, and DeCUR showed the smallest drop (−4.74 pp).\[30\] Interpretation: for unseen Himalayan scenes, domain robustness (augmentation, terrain masking, ensembling) matters more than picking the biggest FM.
3. **SAR-only flood performance is much lower than optical, and debris is not water.** Kaushik et al. (arXiv 2511.01990) found Prithvi best on Sentinel-1 at only 0.57 mIoU, versus Clay 0.72 on Sentinel-2 and 0.79 on PlanetScope. Clay (26M parameters) ran about 3× faster than Prithvi (650M).\[31\] GEOID-Flood (arXiv 2608.02315, 219 CEMS-derived events in 65 countries) found "foundation models offer a consistent but modest advantage". Its best flooded-water IoU was 0.521 with finetuned early S1+S2 fusion, and Kuro Siwo "transfers best on binary water".\[32\]\[33\]

### Ranked model table: flood / water / debris segmentation

| Rank (for this task) | Model | Type | Inputs | Best reported numbers | Weights / license | Fit for Rasuwa |
|---|---|---|---|---|---|---|
| 1 | UNet-ResNet50 (Kuro Siwo BlackBench) | CNN segmentation, multi-temporal stacked | S1 GRD VV/VH, 2–3 dates (pre, pre, post) | F1-F 80.12, mIoU 76.20 (GRD) | Train yourself on Kuro Siwo (allowed) | **Core member.** Fast, fully compliant, multi-temporal |
| 2 | TerraMind v1-B/L (IBM/ESA) | Any-to-any generative multimodal FM | S1, S2, DEM, LULC, NDVI, etc. | Sen1Floods11 mIoU 90.62 (S1+S2); ≥3 pp avg mIoU over other GeoFMs on PANGAEA | Open on HuggingFace, TerraTorch configs for Sen1Floods11 | **Core member (if pretrained weights cleared).** S1+S2 fusion; "Thinking-in-Modalities" can synthesize missing modalities |
| 3 | Prithvi-EO-2.0 300M/600M(-TL) | ViT MAE FM, multi-temporal, time/location embeddings | HLS optical (6 bands); S1 via adaptation | Sen1Floods11 mIoU 89.7–90.5, water IoU 82.2–83.2 | Apache-2.0 (HF) | Strong optical member when S2 is clear; weaker on SAR |
| 4 | Prithvi-CAFE | Prithvi + CNN adaptive fusion encoder | S2 (Sen1Floods11) | mIoU 90.50; Bolivia hold-out 88.87 | Paper code (verify) | Good generalization evidence; small (45.5M trainable) |
| 5 | TerraFlow | Multimodal multi-temporal FM w/ temporal attention | S1 pre/post (+ others) | Reported > UNet-ResNet50 on Kuro Siwo mIoU and F1-F | Check release | Candidate challenger for the Kuro Siwo slot |
| 6 | Galileo (ICML 2025) | Multimodal transformer, global+local SSL | S2, S1, elevation, weather, pseudo-labels, time series | "Outperforms SoTA... across eleven benchmarks" | Open (verify) | Natural fit for S1+S2+DEM+weather fusion; test empirically |
| 7 | Clay v1/v1.5 | MAE ViT FM, sensor-agnostic | S2, S1, Landsat, NAIP, etc. | S2 flood mIoU 0.72, PlanetScope 0.79 (Kaushik et al.) | Open (verify) | Cheap, fast; good S2 member |
| 8 | DOFA | Wavelength-conditioned dynamic FM | Any spectral sensor | Competitive within 2–5% of best GFM (Kaushik et al.) | Open (verify) | Useful if adding NISAR/EOS-04 later |
| 9 | DeCUR / SSL4EO-S12 (MoCo/DINO/MAE) / SoftCon / CROMA | SAR-optical contrastive SSL backbones | S1+S2 | DeCUR smallest geographic drop (−4.74 pp) in PEFT study | Open (verify per repo) | Robustness-oriented backbones for ensemble diversity |
| 10 | FloodViT (Kuro Siwo) | MAE-pretrained ViT on unlabeled Kuro Siwo SAR | S1 time series | F1-F 78.70, mIoU 74.22 | Kuro Siwo repo | SAR-native pretraining from allowed data, so lowest compliance risk among ViTs |
| 11 | ChangeFormer / SNUNet-CD / BIT / TinyCD / STANet | Bi-temporal change detection | Pre/post pairs | ChangeFormer F1-F 76.96; SNUNet-CD 74.26 on Kuro Siwo (BIT not benchmarked there) | Open | Use for **debris/scour change**, not water |
| 12 | Microsoft global SAR flood model (arXiv 2411.01411) | U-Net on 10 yrs S1 | S1 | F1 0.77 on Kuro Siwo vs GFM 0.72; test IoU 0.67 | Outputs, not necessarily weights | Benchmark reference only |
| — | SkySense, AnySat, Copernicus-FM, SatMAE, RemoteCLIP/GeoChat, SAM/SAM2 adapters | FMs / VLMs | Various | No flood numbers in this research comparable to the above | Mixed; SkySense weights restricted | VLMs for sitrep Q&A over map tiles (plug-in); SAM2 for bridge/building polygon refinement on VHR (plug-in) |

### Ranked model table: building / infrastructure damage and landslide/debris

| Rank | Model | Inputs | Numbers | Use here |
|---|---|---|---|---|
| 1 | ChangeMamba / MambaBDA (arXiv 2404.03425) | Pre/post VHR optical | xBD overall F1 81.41%; +4.39 pp over DamFormer\[34\]\[35\] | Plug-in (needs VHR; xBD not an allowed training set) |
| 2 | STSF-Net (arXiv 2604.05527) | Pre-optical + post-SAR (BRIGHT) | F1-clf 75.83%, mIoU 67.91%, destroyed IoU 56.80% | Plug-in for commercial SAR (Capella/ICEYE/Umbra) |
| 3 | 2026 BRIGHT Challenge winners (arXiv 2607.22746) | Pre-optical + post-SAR, instance level | Winners "favored modality-specific encoding, staged or late optical–SAR fusion" | Design pattern: late fusion |
| 4 | Flood-DamageSense (arXiv 2506.06667) | SAR multimodal Mamba | Building-level major-damage F1 0.4610\[36\]\[37\] | Shows the honest ceiling of SAR-only building damage |
| 5 | Insta-BDA | FM fusion | ChangeMamba undercounts buildings (−47.06% count deviation) vs −34.96%\[38\] | Lesson: pixel metrics hide counting error, so report counts with intervals |
| 6 | Multi-encoder ViT S1+S2 landslide (arXiv 2604.05959) | S1+S2 | F1 0.919 (patch classification, not pixel)\[39\] | Debris-patch classifier pattern |
| 7 | ResU-Net (Landslide4Sense) / heterogeneous ensemble (arXiv 2310.05959) | S2 + DEM/slope (+S1) | Ensemble F1 0.69 with S1+S2 vs 0.61 S2-only\[40\] | Debris class pseudo-label refinement (plug-in for training; core-compatible as architecture) |
| 8 | xView2 baseline | Pre/post VHR | Weighted F1 0.2654\[41\] | Historical floor; shows how far the field moved |

**Key interpretation for damage:** at 10 m Sentinel resolution you cannot grade individual Nepali houses reliably. The honest core product is "building footprint intersected by high-confidence debris/flood/scour evidence" (exposure-hit), not "destroyed vs damaged". Flood-DamageSense's building-level major-damage F1 of 0.46 even with SAR-specialized models confirms this.

### 4. Classical and hybrid methods that must be in the core

- **GFM's ensemble is the operational template.** GFM runs three independent algorithms (LIST HASARD, DLR, TU Wien Bayesian) and labels a pixel flooded when "at least two of the algorithms" agree. It applies an Exclusion Mask (radar shadow, strong topography, water look-alikes) and excludes HAND > 15 m, with products "within less than eight hours after image acquisition".\[42\]\[43\] Copy the structure (diverse members, majority/likelihood fusion, explicit exclusion layer), not the outputs.
- **Members to implement:** (1) log-ratio change detection on same-track γ⁰ RTC pairs; (2) split-based automatic thresholding (Martinis-style tiles, bimodality test, then Otsu/KI); (3) Bayesian per-pixel classification against a multi-year same-orbit backscatter climatology (TU Wien style); (4) **coherence-loss damage proxy** following ARIA DPM logic. That means pre-event coherence (γ_pre-pre) minus co-event coherence (γ_pre-post), histogram-matched, thresholded.\[44\]\[45\] ARIA warns DPMs "may be less reliable over vegetated and mountainous areas".\[46\] A 2026 Sentinel-1 time-series study reports coherent change detection reaching F1 69–73% in built-up areas and mapping about 5× more inundation than GFM in some areas.\[47\]
- **Terrain handling is non-negotiable in the Himalaya:** gamma-naught RTC terrain flattening with Copernicus DEM; layover/shadow masks from local incidence angle; ascending + descending fusion to fill each other's shadow zones; and a HAND layer computed from Copernicus DEM (allowed input) rather than MERIT. Do **not** blindly apply GFM's 15 m HAND cut. The Trishuli rose 9 m, and debris surges run up terraces well above normal flood levels, so use a HAND threshold scaled by flow-depth estimates along the traced path.
- **Narrow gorges break CNNs.** Konapala et al. found that CNN-based segmentation "may fail to correctly classify pixels around narrow river channels".\[48\] Pair the ensemble with a channel-aware post-process: the flow-path corridor from DEM routing plus a minimum-mapping-unit relaxation inside the corridor.
- **Optical indices when clear:** MNDWI/NDWI for water; NDVI loss plus BSI (bare soil index) increase for debris and scour; SWIR-based turbidity for sediment-laden water. Cloud masking via the S2 SCL band and s2cloudless (both derivable from allowed S2 data). Cloud Score+ is a GEE product derived from S2; treat it as a derived convenience, flagged.

### 5. Cloud handling and SAR-to-optical reconstruction: use carefully

SAR-assisted cloud removal on SEN12MS-CR has reached roughly 31–33 dB PSNR: DSen2-CR 27.76 dB; GLF-CR 29.08 dB; a cloud-aware attention method 31.01 dB / SSIM 0.918; TFCRNet 33.01 dB.\[49\]\[50\]\[51\]\[52\] Diffusion variants (DB-CR, CM-CR, SAR-DeCR) claim SOTA.\[53\]\[54\]\[55\] **Recommendation:** use reconstructed optical only for *visualization and analyst context*, never as primary evidence for a damage claim. Generative models can hallucinate plausible terrain. GLF-CR's own analysis shows performance degrades as the time interval to the reference grows (PSNR 28.65 → 25.31 dB across longer intervals).\[51\] Debris deposition is precisely the kind of abrupt change a generator will smooth away. TerraMind's "Thinking-in-Modalities" (generating synthetic S2-like tokens from S1) is a more principled use: it improves the classifier internally without fabricating a user-facing image.\[56\]

### 6. Datasets (ranked for Himalayan debris-flood relevance)

| Rank | Dataset | Modality | Scale / content | License | Compliance status | Relevance |
|---|---|---|---|---|---|---|
| 1 | Kuro Siwo (NeurIPS 2024) | S1 GRD + SLC, DEM, multi-temporal | 43 events, 67,490 time series, 202,470 SAR samples, 224×224 tiles; plus large unlabeled SAR set | Verify (repo) | **Allowed (recommended)** | Primary SAR training; SLC enables coherence |
| 2 | Sen1Floods11 | S1 + S2 + labels | 4,831 chips 512×512, 11 events, 120,406 km²; 446 hand-labeled | Verify (CC BY 4.0 per common usage) | **Allowed (optional)** | S1+S2 fusion training; Bolivia hold-out |
| 3 | GEOID-Flood (2026) | bi-temporal S1 + S2 + DEM | 219 events, 65 countries, CEMS-derived | Verify | Not allowed for training (and CEMS-derived labels) | Validation/plug-in only |
| 4 | Landslide4Sense | S2 + ALOS DEM/slope | 3,799 train patches, 14 bands, ~2% positives\[57\]\[58\] | Verify | Not listed | Debris class (plug-in training) |
| 5 | BRIGHT | VHR pre-optical + post-SAR | ~4,200 pairs, 380,000+ buildings, 14 regions | CC BY-SA 4.0 | Not listed | Building damage plug-in |
| 6 | xBD / xView2 | VHR optical pre/post | >700,000 buildings, 6 disaster types | Non-commercial (verify) | Not listed | Plug-in |
| 7 | UrbanSARFloods, S1GFloods, ETCI 2021, MMFlood, S1S2-Water, WorldFloods v2, SEN12-FLOOD, UFO (Urban Flood Observations), Sen2GF3Floods | S1 and/or S2 | Various; UFO model IoU 76.1–82.2 vs S1-IMPACT 44.1 | Mixed, verify each | Not listed | Cross-dataset stress testing (plug-in) |
| 8 | SpaceNet 8 | VHR optical | Flooded roads + buildings | Verify | Not listed | Road-flood plug-in |
| 9 | HR-GLDD, ICIMOD glacial lake inventories, GLOF databases | Optical/inventory | Glacial lakes HKH | Verify | Not listed | Proactive layer only |
| 10 | Global Flood Database, DFO archive, NASA Global Landslide Catalog | Event archives | Historical | Open (verify) | Not listed | Historical recurrence layer (plug-in) |

A sobering calibration point from the Kuro Siwo paper: for one event, CEMS and Kuro Siwo expert annotations agreed at only 51% (permanent water) and 48% (flood) IoU.\[25\] **Reference maps are not ground truth.** Your EMSR927 comparison should be framed as agreement with a different expert product, with that ceiling stated.

### 7. Agency data and services (plug-in layer unless noted)

| Source | What it gives | Resolution / latency | Access | Plug-in role |
|---|---|---|---|---|
| **Sentinel-1C/1D** (core) | C-band SAR GRD/SLC | 10 m; 6-day revisit restored after Sentinel-1A ended operations 29 June 2026; C/D shift the old pattern by 1 day\[59\]\[60\] | CDSE, ASF, Planetary Computer | Core evidence, same-track pairs |
| **Sentinel-2** (core) | MSI optical | 10–20 m; 5-day revisit | CDSE, AWS, PC | Debris/water when clear |
| **Copernicus DEM GLO-30** (core) | DSM | 30 m | Open | HAND, flow routing, slope, LIA, HLZ |
| **NISAR** (NASA-ISRO) | L-band (and S-band) SAR; GCOV RTC γ⁰ at 10/20/80 m, InSAR products | Provisional calibrated L-band public since 20 July 2026 for acquisitions from 17 June 2026; L0B latency 2–10 h; **permanent instrument gap 27 Jul–10 Aug 2026** | ASF DAAC / Earthdata; S-band via ISRO Bhoonidhi | Better coherence over vegetation/steep slopes; but the gap removes the nearest pre-event baseline for this event |
| **ISRO EOS-04 (RISAT-1A)** | C-band SAR | Tasked | Bhoonidhi / NRSC | Revisit-gap filler |
| **Bhuvan / NDEM / NRSC DSC** | Indian disaster portals, flood layers | Event-based | Bhuvan (Indian territory focus) | Reference, cross-border India extension |
| **OPERA DSWx-S1 / DIST-S1 / DIST-ALERT-HLS / RTC-S1 / CSLC / DISP** | 30 m water, disturbance alerts, ARD | DSWx-S1 since Sept 2024; DIST-S1 record from March 2026; DIST-ALERT-HLS can update alerts within about six hours\[61\]\[62\] | ASF / PO.DAAC / LP DAAC | RTC-S1/CSLC are ARD from S1 (gray area); DSWx/DIST are published maps (validation only) |
| **ARIA DPM** | Coherence-based damage proxies | 30 m, event-based\[63\] | NASA Disasters | Validation / method reference |
| **GPM IMERG** | Rainfall | ~0.1°, NRT runs | NASA | Proactive trigger; also proves "no rain" (dry-weather surge → cryosphere trigger) |
| **MODIS/VIIRS NRT flood (LANCE)** | Daily coarse flood | 250–375 m | NASA | Coarse corroboration |
| **ICESat-2, GRACE** | Altimetry, mass | Sparse | NASA | Glacier elevation change (proactive research only) |
| **GFM / GloFAS** | Ensemble S1 floods; forecasts | 20 m flood maps, <8 h | CEMS | Validation only (published flood maps) |
| **CEMS Rapid Mapping** | FEP/DEL/GRA | On request, hours–days | CEMS | Validation only |
| **ICIMOD / SERVIR-HKH** | Glacial lakes, HKH hazard science | Event hubs | ICIMOD RDS | Proactive GLOF layer |
| **Nepal DHM / BIPAD** | Gauges, SMS warnings, incident portal | Real-time when stations survive | Government | IoT/gauge fusion; sitrep cross-check |
| **Planet, Capella, ICEYE, Umbra, Vantor** | VHR optical/SAR | Sub-meter, tasked | Commercial / disaster programs | Damage grading plug-in (BRIGHT-style) |
| **Google Flood Hub** | Riverine ML forecasts | Daily | Public | Not designed for cryosphere-triggered surges; context only |
| **Planetary Computer / GEE / openEO / Sentinel Hub** | Compute + catalogs | — | — | Execution backends via adapters |

### 8. Existing systems: features, strengths, gaps

| System | Strength | Gap exposed by Rasuwa 2026 |
|---|---|---|
| CEMS Rapid Mapping (EMSR927) | Expert VHR grading, standardized products | AOI-limited; early hours had no products; two AOIs "not feasible"; no network analysis |
| International Charter (1052) / Sentinel Asia | Free multi-agency imagery fast | Delivers imagery/products, not decisions; cloud-limited optical |
| GFM | Fully automatic, global, ensemble, <8 h | HAND>15 m and topography exclusion masks remove much of Himalayan valley sides by design; water-only, not debris\[64\] |
| Google Flood Hub | ML riverine forecasting | Forecasts rain-driven floods; a dry-weather ice-rock avalanche surge is out of scope |
| HOT Tasking Manager + fAIr | Crowd + AI building mapping, fast (27–28 Aug)\[21\] | Post-event OSM edits are disallowed inputs for you; quality varies |
| UNOSAT | Expert analysis (Sentinel-2-based post-event mapping) | Published damage maps, disallowed as inputs |
| NASA Disasters / ARIA | DPMs, OPERA | DPM unreliable in mountains/vegetation\[46\] |
| Nepal DHM / BIPAD | SMS early warning, gauges | Four stations destroyed;\[9\] DHM informed by phone at 09:00 |
| ICIMOD | Science hub, GLOF inventories | Analysis, not operational routing |
| Floodbase, Cloud to Street, ICEYE flood | Commercial parametric/flood products | Commercial; riverine/pluvial focus |

### 9. Latest arXiv / journal papers (2024–2026): one-line takeaways

- **Kuro Siwo (arXiv 2311.12056, NeurIPS 2024):** multi-temporal GRD segmentation beats change detection; UNet-ResNet50 F1-F 80.12.\[25\] Use as your SAR backbone recipe.
- **Prithvi-EO-2.0 (arXiv 2412.02732):** TL (time/location) 600M variant reaches Sen1Floods11 mIoU 90.5.\[28\] Metadata embeddings help modestly.
- **TerraMind (arXiv 2504.11171):** first any-to-any EO FM; S1+S2 beats either alone on Sen1Floods11 (90.62 vs 89.57/80.39).\[27\]\[56\]
- **PANGAEA (arXiv 2412.04204):** standard GFM benchmark; most GFMs did not beat U-Net across tasks. Don't assume FM > CNN.
- **PEFT for GFMs (arXiv 2504.17397):** 7.31 pp average geographic drop on Sen1Floods11;\[30\] LoRA-style PEFT competitive. Fine-tune cheaply, evaluate on held-out regions.
- **Kaushik et al. (arXiv 2511.01990):** GFMs give 2–5% spreads; Clay fastest.\[31\] Model choice matters less than sensor and site.
- **Prithvi-CAFE (arXiv 2601.02315):** CNN+FM fusion lifts Bolivia hold-out to 88.87 mIoU.\[29\] Hybrid encoders generalize.
- **TerraFlow (arXiv 2603.12762):** temporal attention beats BlackBench baseline on Kuro Siwo.\[26\] Model time explicitly.
- **GEOID-Flood (arXiv 2608.02315):** FM advantage "consistent but modest"; fusion helps transient floods.\[33\] Fuse S1+S2 when both exist.
- **Microsoft 10-year SAR floods (arXiv 2411.01411):** F1 0.77 on Kuro Siwo vs GFM 0.72.\[65\] Learned models beat threshold ensembles on average.
- **Understanding flood detection across S1/S2 (RSE 2025):** multi-dataset multimodal model is most consistent (83.55 IoU on UNOSAT set).\[66\] Cross-dataset training > single-dataset.
- **Urban Flood Observations (arXiv 2604.23066):** leave-one-event-out mIoU 77.3%; operational S1 products had high precision but recall only 46.2%.\[67\] Operational systems under-detect.
- **GFM scientific review (RSE 2025):** HAND priors reduce false negatives but raise false positives; underdetection remains.\[64\] Your physics masks need tuning, not hard cuts.
- **Coherent + incoherent CD (2026):** F1 69–73% for built-up flood/damage proxy.\[47\] Add coherence.
- **BRIGHT (ESSD 2025) and BRIGHT Challenge 2026 (arXiv 2607.22746):** pre-optical + post-SAR is the all-weather recipe; cross-event generalization is the unsolved problem.\[68\]
- **ChangeMamba (arXiv 2404.03425):** SSM change detection SOTA on xBD (81.41 overall F1).\[35\]
- **Flood-DamageSense (arXiv 2506.06667):** SAR-only building damage remains hard (major-damage F1 0.46).\[37\]
- **Multi-encoder ViT landslide (arXiv 2604.05959):** S1+S2 fusion F1 0.919 at patch level, without pre-event S2.\[39\]
- **Clay-CNN hybrids for Landslide4Sense (arXiv 2606.14081):** FM context helps; calibrated uncertainty rises on errors (σ_FP +67%).\[57\] Uncertainty is informative.
- **Chamoli pre-collapse motion (NHESS 2022):** Sentinel-2 feature tracking revealed ~20–30 m displacement 2016–2021 before the 2021 rock–ice avalanche, while Sentinel-1 InSAR "would not have provided an adequate tool".\[69\]\[70\] Offset tracking, not InSAR, for proactive glacier-slope watch.
- **SAR cloud removal (GLF-CR, DB-CR, CM-CR, SAR-DeCR, 2022–2025):** ~29–33 dB PSNR on SEN12MS-CR.\[51\]\[52\] Good visuals, risky evidence.
- **DeepSAR Flood Mapper (2025) / MSFlood-Net:** S1+HAND inputs improve physical consistency.\[71\]\[72\] Feed HAND as a channel, not only as a mask.

## Details

### A. Competition-compliant core architecture

**Inputs (only):** Sentinel-1 GRD + SLC (same relative orbit, both ascending and descending if available); Sentinel-2 L2A; Copernicus DEM GLO-30; OSM via ohsome API with snapshot timestamp ≤ 2026-07-27; Kuro Siwo and Sen1Floods11 for training.

**Stage 0: AOI and date resolver.** Given an area and flood date, auto-select: the last two pre-event S1 acquisitions on each relative orbit and the first post-event acquisition per orbit (Kuro Siwo-style triplets); and the clearest pre/post S2 within ±20 days, using SCL/s2cloudless cloud fraction over the AOI, not the scene. Log every scene ID, so provenance is automatic.

**Stage 1: SAR ARD.** Apply orbit files, thermal noise removal, calibration to β⁰, then terrain flattening to γ⁰ (RTC) with Copernicus DEM, and geocoding. Compute local incidence angle; build layover/shadow masks; apply refined Lee or multitemporal speckle filtering. For SLC: produce coherence for pre-pre and pre-post pairs (6- or 12-day baselines).

**Stage 2: terrain physics layers (precomputed, cacheable).** Hydrologically conditioned DEM, D8/D∞ flow direction and accumulation, drainage network, HAND, slope, and a valley-confinement index. Use pysheds/richdem/WhiteboxTools.

**Stage 3: ensemble evidence (per pixel, per class: water / flood / debris-scour / no-change).**
- M1: UNet-ResNet50 on Kuro Siwo GRD triplets (water/flood).
- M2: TerraMind or Prithvi-EO-2.0 fine-tuned on Sen1Floods11 (S1+S2 when S2 clear; S1-only head otherwise).
- M3: classical members: log-ratio + split-based Otsu; Bayesian against same-orbit climatology; coherence-loss DPM.
- M4: debris-scour change member: bidirectional |log-ratio| (both darkening and brightening), coherence loss, texture change (GLCM), and, when clear, S2 NDVI-loss + BSI-gain. Trained or self-trained as below.
- Fusion: calibrated likelihoods (temperature scaling per member on held-out Kuro Siwo events), combined by weighted log-opinion pooling. Add a GFM-style "at least 2 of N agree" rule for binary maps, plus an exclusion layer (layover/shadow, LIA extremes, HAND above the path-adaptive limit, permanent water from the pre-event S1/S2 stack).

**Stage 4: physics-informed correction.** Constrain flood/debris to the hydraulically connected corridor: pixels must be reachable via downslope flow paths from the traced channel. HAND must fall below a reach-specific depth envelope. Isolated positives on high HAND/steep slope are down-weighted, not deleted (keep them as "unexplained change" for analyst review). This is the physics-guided prior, applied as a Bayesian prior map, not a hard mask.

**Stage 5: damage overlay.** For each pre-event OSM building, road segment and bridge, compute: fraction of footprint/buffer in each class, max likelihood, and evidence-source agreement.
- **Bridges get a dedicated test.** A bridge is a strong, stable radar scatterer (double-bounce with water). Loss of the bright point-target signature between pre and post γ⁰ at the bridge geometry, combined with debris/flood evidence at the abutments, raises P(bridge lost). This is the single most important damage variable for cut-off analysis.
- **Outputs are probabilities with confidence tiers,** not binary "damaged".

**Stage 6: probabilistic isolation engine (the novel core).**
- **Graph.** OSM highway=* including track, path, footway and steps, because trails matter in Rasuwa. Bridge ways are edges with their own failure probabilities. Ferries and suspension bridges are tagged where present.
- **Targets.** Nearest hospital (amenity=hospital, healthcare=*), district HQ (Dhunche for Rasuwa; Bidur for Nuwakot), and the nearest town outside the impact zone.
- **Monte Carlo.** For N=1,000–10,000 draws, sample each edge's state from its failure probability (with spatial correlation along a reach). Compute reachability (networkx/igraph multi-source BFS on the reverse graph; osmnx for graph build). Each settlement gets P(cut off), expected shortest-path increase, and its "critical edge set" (edges whose loss most often disconnects it).
- **Ranking.** Priority score = P(cut-off) × exposure proxy (pre-event OSM building count in the settlement polygon, since population rasters are not allowed inputs) × hazard proximity × (1 + distance to the nearest surviving HLZ candidate).
- **HLZ candidates.** Flat cells from Copernicus DEM slope (e.g., ≤7° over a contiguous clear patch of at least ~50 m), outside the debris/flood corridor and HAND-safe. Exclude OSM buildings, power lines and forest landuse, and prefer approach corridors free of steep relief. Copernicus DEM at 30 m cannot see small obstacles, so label these "candidate, needs aircrew confirmation".

**Stage 7: bonus flow tracing.** From any user-clicked upstream point, trace the D8 steepest-descent path on the conditioned DEM to the AOI outlet. Buffer it by a depth-envelope-derived width and list settlements in order with along-channel distance. Estimate arrival times with a reach-segmented celerity model calibrated on this event's anchors: ~193 km/h over the first 22 km (Kargel); Rasuwagadhi impact around 08:44 (per a third-party event site); Devghat arrival ~15:20 and peak 16:00 (DHM-derived timeline).\[5\]\[10\]\[20\] Report celerity uncertainty bands, not single times.

**Stage 8: grounded sitrep.** All numbers come from SQL views over the result database (PostGIS). The LLM receives a JSON "fact sheet" of IDs, values, units, timestamps and confidence tiers. It writes English and Nepali prose using placeholders (e.g., {N_SETTLEMENTS_CUTOFF_HIGH}). A verifier then (1) parses every numeral in the output, including Devanagari digits ०–९; (2) checks each against the fact sheet; (3) rejects or regenerates on mismatch; (4) appends a provenance footer with scene IDs and processing timestamp. Nepali output is template-first (pre-translated fixed phrases reviewed by a native speaker), with LLM only for connective prose.

### B. Why choose AI option (a), and how to prove Himalayan generalization

The 30% mapping score and 20% cut-off score both depend on segmentation quality. Option (a) therefore compounds into 75% of the rubric, while option (b) competes only within the 25%. You can still deliver a robust sitrep via deterministic templating.

**Generalization protocol (empirical model ranking test plan):**
1. **Leave-region-out splits on Kuro Siwo and Sen1Floods11.** Hold out all mountainous/high-relief events (rank by DEM relief) as a "pseudo-Himalaya" test fold. Report F1-flood and mIoU per member and for the ensemble.
2. **Unseen Himalayan scenes:** run zero-shot on Rasuwa 2026, and on other Himalayan events processed from raw S1/S2 (e.g., Chamoli Feb 2021; Melamchi June 2021; the 8 July 2025 Lhende/Purepu GLOF on the same channel). Score against your own clear-sky Sentinel-2 interpretation (allowed input) as an independent check. Use EMSR927 only in the sealed validation notebook.
3. **Metrics:** pixel IoU/F1 per class; building-level precision/recall for "hit"; edge-level bridge-loss accuracy against DoR/news-reported bridge lists (checking only); calibration (ECE, reliability diagrams); and coverage (fraction of AOI not in exclusion mask).
4. **Ablations:** ±DEM/HAND channel; ±coherence; ±S2; ±physics prior; single-member vs ensemble; TTA on/off.
5. **Statistical ranking:** bootstrap over events (not pixels) for 95% CIs; rank members by event-level mean F1 with Friedman/Nemenyi tests.

**Fine-tuning strategy:** PEFT (LoRA) on the FM members; full fine-tune for the CNN. Use Lovász/Dice + weighted CE for class imbalance. Augment with LIA-dependent radiometric jitter, speckle re-simulation, random layover masking, and elevation-shift augmentation. Use test-time augmentation (flips/rotations) and MC-dropout or deep-ensemble variance for epistemic uncertainty.

**Debris class without debris labels (compliance-aware):** neither allowed dataset labels debris. Run test-time self-training: on the target AOI, create high-confidence pseudo-labels from agreement between the S2 NDVI-loss/BSI rule and the SAR bidirectional change + coherence-loss rule. Then adapt only the M4 head on that scene. This uses only allowed inputs, but it is "training on the evaluation scene". Disclose it explicitly and ask organizers whether test-time adaptation is acceptable.

### C. Extended / plug-in layer

**Architecture principles**
- **STAC everywhere:** every source is an adapter emitting STAC Items (pystac); rasters are COGs; vectors are GeoParquet/FlatGeobuf.
- **Compute backends behind one interface:** local (rasterio/xarray/dask), openEO, Sentinel Hub, Planetary Computer, GEE.
- **Model registry** (MLflow-style) with a model card, training data lineage, license and compliance tag ("core-allowed", "plug-in only").
- **Ensemble layer** reads any registered member's calibrated likelihoods.
- **Event bus** (Kafka/NATS/Redis Streams) with topics such as new_scene, gauge_reading, seismic_event, news_item and product_ready.
- **IoT** via MQTT to a time-series DB.
- **OGC outputs:** WMS/WMTS for tiles, OGC API Features/Tiles/Processes for analytics, and CAP 1.2 XML for alerts to DHM/BIPAD SMS gateways.
- **Dashboard:** MapLibre with a PostGIS/pg_tileserv backend.

**Latency budget (target, S1 path):** S1 availability (hours; external) → ARD 10–20 min per AOI on GPU/CPU cluster → ensemble inference 2–5 min → physics/fusion 2 min → network Monte Carlo 1–3 min → sitrep 1 min. Internal processing under 30 minutes after download. Overall time-to-product is dominated by satellite revisit and downlink, which is why the proactive layer matters.

**Plug-in sources and how they help**
- **NISAR L-band.** Higher coherence over vegetated steep slopes, so better debris and landslide coherence-loss detection. Note the permanent data gap 27 Jul–10 Aug 2026, which means the nearest pre-event NISAR baseline for this event is pre-27 July.\[73\]
- **EOS-04 and commercial SAR (ICEYE/Capella/Umbra).** Tasked revisit fillers. Plug in BRIGHT-style pre-optical/post-SAR building damage models (STSF-Net, challenge-winner late fusion).
- **OPERA RTC-S1/CSLC** as ready ARD (saves ARD stage), and **DSWx-S1/DIST-ALERT-S1** as independent corroboration layers.
- **GPM IMERG.** Distinguishes rain-driven floods from dry-weather cryosphere surges (no rain at Dhunche that morning).\[7\]\[74\] The proactive layer's trigger classifier uses this.
- **Gauges/IoT.** The four destroyed stations show sensors fail; design for the "sensor went silent" signal itself as an alarm, and fuse stage jumps (9 m in 30 min) as event detection.
- **Seismic.** USGS/GFZ detection of a landslide-generated M5.2-equivalent signal.\[4\]\[75\] A regional seismic-to-CAP rule would have fired minutes after 08:40.
- **News/social media NLP.** Fine-tune on CrisisNLP/HumAID-style taxonomies for "infrastructure damage", "missing people" and "requests". Geoparse with a Nepali gazetteer built from the pre-event OSM place names (with Devanagari/romanization variants). Every extracted claim becomes a "report" object with source, time and confidence, shown alongside satellite evidence, never merged into satellite counts.
- **VLM Q&A.** RemoteCLIP/GeoChat-style models answering "is the bridge at X visible?" on VHR chips. Strictly advisory.

**Can non-listed datasets be used for pretraining/validation?**
- **Validation (checking only):** the rules explicitly allow CEMS/UNOSAT/post-event OSM "checking only". Keep these in a separate repo folder and notebook, never imported by the pipeline code. Add a CI test that fails if any pipeline module references those paths or domains.
- **Pretraining:** foundation-model weights pretrained on non-listed data (Prithvi on HLS, TerraMind on its multimodal corpus) are not "inputs" at inference, but they are learned from non-listed data. **Risk: medium.** Mitigation: ask organizers in writing; keep a fully compliant fallback (UNet-ResNet50 and FloodViT pretrained only on Kuro Siwo's unlabeled SAR); report both.
- **Training on BRIGHT/xBD/Landslide4Sense:** **high risk**, as these are explicitly not on the list. Plug-in layer only.
- **Auxiliary global layers** (JRC Global Surface Water, ESA WorldCover, MERIT HAND, WorldPop): **not allowed** as core inputs. Derive equivalents from S1/S2/DEM/OSM instead.

### D. Proactive layer design (beyond revisit limits)

1. **Glacier/slope watch.** Monthly Sentinel-2 and Sentinel-1 offset tracking (autoRIFT/GIV) over glaciers and steep rock walls above settlements. Chamoli shows optical feature tracking detected multi-year acceleration where InSAR failed. Add NISAR L-band where coherent. Flag accelerating blocks with runout envelopes.
2. **Runout pre-simulation.** For flagged sources and glacial lakes (ICIMOD inventories), precompute r.avaflow (multi-phase mass flow) or simpler Voellmy/flow-R envelopes. Also run HEC-RAS 2D/LISFLOOD-FP dam-break scenarios for valley reaches. Store inundation/arrival-time envelopes per settlement, so the system can issue a warning from the trigger alone.
3. **Trigger fusion.** Seismic detection of landslide-type signals + gauge jump or "gauge silence" + no IMERG rain → classify as a cryosphere mass-flow surge → issue a CAP alert to downstream settlements ordered by precomputed arrival time. Using this event's anchors, upper-corridor settlements near the source had only minutes. But Betrawati/Trishuli/Nuwakot and the Narayani-Devghat reach had on the order of hours (Devghat arrival ~6.5 h after the signal).\[10\] That downstream window is where lives are saved.
4. **Pre-event criticality ("which bridges matter most").** On the pre-event OSM graph, compute for every bridge the number of settlements and buildings that become unreachable if it alone fails (single-edge cut analysis), plus pairwise failures along a reach. Publish this as a peacetime planning layer: where to preposition Bailey bridges (the government ultimately requested 42), food stocks, and certified HLZs.\[12\]
5. **Automatic tasking.** On trigger, auto-generate Charter/Sentinel Asia request text and commercial SAR tasking polygons along the traced path, prioritized by the criticality layer.

### E. Historical event mapping and risk scoring

Back-process the full Sentinel-1 archive (2014–present) along each Himalayan corridor with the same pipeline to build per-road-segment and per-settlement recurrence counts (events hit, years, max class). Join with the Global Flood Database, DFO archive and NASA Global Landslide Catalog points (plug-in). Do this especially for the Lhende–Bhotekoshi channel, which also flooded from the Purepu glacier lake on 8 July 2025, 36 km north of the border.\[8\] Risk score per segment = recurrence × criticality × exposure. This turns one-off response maps into a living hazard register.

### F. "Military-grade" product standards, reproduced with open data

- **Confidence language:** adopt intelligence-style confidence tiers (High/Moderate/Low), tied to quantitative rules, e.g., High = ≥3 independent members agree, calibrated p ≥ 0.8, outside the exclusion mask, and corroborated by a second orbit or S2.
- **Provenance:** every polygon carries scene IDs, model versions, git commit and timestamps (W3C PROV-style JSON).
- **Change timeline:** per object, "last seen intact (date, sensor)" and "first seen changed (date, sensor)".
- **LOC analysis:** lines of communication as graph edges with status (open / probably cut / cut / unknown) and chokepoints (bridges, gorge roads, single-access trails).
- **Isochrones** (walking and driving) from surviving nodes to hospitals.
- **HLZ list** with slope, size, obstacle notes and approach direction.
- **Multi-source corroboration matrix** per claim (SAR, optical, physics, reports).
- **Explicit "unknown" class:** shadow/layover zones are displayed as "not observable", never as "no damage". That alone exceeds most public products.

### G. Validation against EMSR927

EMSR927 consists of GRA products over AOIs (Syapru Besi, Timure, Bidur, Phosretar per the third-party tracker), produced from VHR optical (WorldView-3, Legion, Pléiades Neo, etc.).\[24\] Compare in the sealed notebook:
- **Building level:** your "hit" probability vs CEMS "destroyed/damaged" on matched pre-event OSM buildings. Report precision/recall/F1 and PR curves.
- **Extent level:** IoU of your debris+flood union vs CEMS flood trace/affected area, inside your observable mask only.
- **Infrastructure level:** CEMS-graded roads/bridges vs your edge states.

Report results per AOI. Note that EMS sensors are 0.3–1 m while yours are 10 m: expect strong recall on destroyed clusters and weak recall on isolated damaged houses. Contextualize with the 48–51% inter-annotator IoU observed between CEMS and Kuro Siwo labels.\[25\]

## Recommendations

1. **Build the core in this order:** ARD + terrain layers → UNet-ResNet50 Kuro Siwo member → classical GFM-style members + coherence → debris member → fusion/calibration → bridge test → Monte Carlo isolation → HLZ → flow tracing → grounded bilingual sitrep → dashboard.
2. **Make the headline product "Settlement Isolation Probability"** with a ranked reach-first list and HLZ candidates; it is the clearest answer to what failed (bridges, helipads, 15-day waits).
3. **Submit AI option (a),** with a leave-mountainous-events-out protocol and zero-shot results on Rasuwa, Chamoli and the July 2025 Lhende event.
4. **Request written organizer rulings** on (i) pretrained FM weights and (ii) test-time self-training. Ship a fully compliant fallback either way.
5. **Enforce compliance in code:** an allowlist of data domains in the adapter layer; CI test blocking CEMS/UNOSAT/GFM/OPERA-map/HOT-post-event sources from pipeline imports; ohsome queries pinned to 2026-07-27.
6. **Demo narrative (3 min):** dry-weather signal → seismic/gauge trigger → S1 first-pass map → bridge losses → isolated settlements with probabilities → HLZs → Nepali sitrep with every number traceable → EMSR927 comparison and limitations.

## Caveats

- **Revisit limits are real:** Sentinel-1C/1D gives a 6-day repeat, so there is no minutes-ahead warning of glacier collapse from Sentinel alone; the proactive layer depends on non-core sensors and precomputed models.\[76\]
- **Himalayan SAR pitfalls:** layover/shadow in narrow gorges (often unobservable from one orbit); wet snow and glacier surfaces mimic water darkening; speckle at 10 m; steep LIA radiometric artifacts; vegetation decorrelation that makes coherence-loss noisy (ARIA's own warning). Debris deposits can brighten rather than darken backscatter, so water-only models under-detect them.
- **10 m resolution** cannot grade individual building damage; outputs are exposure-hit probabilities.
- **Copernicus DEM is a 30 m DSM** (includes canopy/buildings), which degrades HAND, routing and HLZ slope in narrow valleys.
- **Pre-event OSM may be incomplete** (missing trails, suspension bridges, unmapped hamlets); isolation results inherit those gaps. Building-count exposure is a proxy for population.
- **Sources disagree.** On location: Nepal side vs Tibet side. On bridges: CARE's early 32 vs the Department of Roads' 41 destroyed plus 4 damaged. On cost: Rs20 billion vs Rs30 billion. On deaths: one NGO reported 1,250 confirmed dead, the problem statement cites 1,342 (NDRRMA, by 6 Sept), and UN OCHA SitRep No. 7 reported "The floods have claimed 1,356 lives. 4,894 people are still missing" (NDRRMA, 7 Sep 2026, 19:00). On mechanism: GLOF vs ice-rock avalanche. Several EMSR927 product details come from a third-party API tracker and an NGO page, not the official CEMS page, and need confirmation.
- **Benchmark numbers** come from heterogeneous splits and class definitions (multi-class Kuro Siwo vs binary elsewhere) and are not directly comparable; some 2026 arXiv results are preprints not yet peer-reviewed.
- **Celerity/arrival-time calibration** is drawn from a single event's reported anchors and should be presented with wide uncertainty bands.

## Sources

1. [Bridges gone, roads damaged, villages cut off from relief](https://kathmandupost.com/province-no-3/2026/09/17/bridges-gone-roads-damaged-villages-cut-off-from-relief)
2. [What You Did for the Animals of Nepal - ROLDA](https://rolda.org/what-you-did-for-the-animals-of-nepal)
3. [How Nepal's Rescue Teams Reached Flood Survivors in Rasuwa](https://travelhimalayanepal.com/news/nepal-army-flood-rescue-rasuwa-2026)
4. [Satellite images show destruction from Nepal-Tibet floods](https://www.aljazeera.com/news/2026/8/27/satellite-images-show-destruction-from-nepal-tibet-floods)
5. [August 26-27, 2026 — Deadly flash flood hits Nepal-China border region](https://www.cnn.com/2026/08/26/world/live-news/nepal-flash-flooding-floods-intl)
6. [2026 Nepal floods](https://en.wikipedia.org/wiki/2026_Nepal_floods)
7. [2026 Nepal–Tibet floods](https://en.wikipedia.org/wiki/2026_Nepal%E2%80%93Tibet_floods)
8. [Kyirong-Rasuwa Flood 2026](https://www.icimod.org/kyirong-rasuwa-flood-2026-nepal-china-border/)
9. [Timeline of the 2026 Nepal floods](https://en.wikipedia.org/wiki/Timeline_of_the_2026_Nepal_floods)
10. [Timeline of the 2026 Nepal–Tibet floods](https://en.wikipedia.org/wiki/Timeline_of_the_2026_Nepal%E2%80%93Tibet_floods)
11. [Nepal flash floods 2026: CARE's emergency response - CARE](https://www.care.org/news-and-stories/nepal-flash-floods-2026-care-emergency-response/)
12. [Washed-out bridges force flood-hit communities to rely on choppers](https://kathmandupost.com/national/2026/08/29/washed-out-bridges-force-flood-hit-communities-to-rely-on-choppers)
13. [Oft-battered Langtang Valley cut off after Rasuwa floods](https://kathmandupost.com/money/2026/09/02/oft-battered-langtang-valley-cut-off-after-rasuwa-floods)
14. [August 2026 Nepal Trishuli Flood](https://storymaps.arcgis.com/stories/f2b2425eac544929a7d18f4c90b41d66)
15. [The Copernicus Emergency Management Service maps flood damages in northern Nepal](https://eu-space.europa.eu/components/earth-observation-copernicus/image-of-the-day/copernicus-emergency-management-service-maps-flood-damages-northern-nepal)
16. [Flood in Nepal - Charter Activations](https://disasterscharter.org/activations/flood-in-nepal-activation-1052-)
17. [Home - Sentinel Asia Web Site](https://sentinel-asia.org/)
18. [NESRA FloodWatch — Rasuwa / Bhotekoshi–Trishuli Flood 2026](https://nesraspace.org/floodwatch/rasuwa-2026/)
19. [Aftermath of Nepal flash flood](https://eu-space.europa.eu/components/earth-observation-copernicus/image-of-the-day/aftermath-nepal-flash-flood)
20. [Rasuwa Flood](https://rasuwaflood.org/)
21. [Organised Editing/Activities/Nepal Floods 2026 - OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Organised_Editing/Activities/Nepal_Floods_2026)
22. [EMSR927: Imagen/sensor actualizado: AOI04 Bharatpur GRA v1 (). · Issue #7 · asoto59g/Nepal](https://github.com/asoto59g/Nepal/issues/7)
23. [EMSR927: Version nueva: AOI05 Phosretar GRA v1 (antes v3). · Issue #9 · asoto59g/Nepal](https://github.com/asoto59g/Nepal/issues/9)
24. [EMSR927: La activacion se cerro. · Issue #11 · asoto59g/Nepal](https://github.com/asoto59g/Nepal/issues/11)
25. <https://arxiv.org/pdf/2311.12056>
26. [TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation](https://arxiv.org/pdf/2603.12762)
27. [TerraMind: Large-Scale Generative Multimodality for Earth Observation](https://arxiv.org/pdf/2504.11171)
28. [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://arxiv.org/pdf/2412.02732)
29. [Prithvi-Complimentary Adaptive Fusion Encoder (CAFE): unlocking full-potential for flood inundation mapping](https://arxiv.org/pdf/2601.02315)
30. [Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models](https://arxiv.org/html/2504.17397v1)
31. [Assessing the value of Geo-Foundational Models for Flood Inundation Mapping: Benchmarking models for Sentinel-1, Sentinel-2, and Planetscope for end-users](https://arxiv.org/pdf/2511.01990)
32. [GEOID-Flood: A Large-Scale Multi-Modal Benchmark Dataset for Flood Segmentation](https://arxiv.org/html/2608.02315)
33. [GEOID-Flood: A Large-Scale Multi-Modal Benchmark Dataset for Flood Segmentation](https://arxiv.org/pdf/2608.02315)
34. [ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model](https://www.alphaxiv.org/abs/2404.03425)
35. [ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model](https://arxiv.org/html/2404.03425v1)
36. [Flood-DamageSense: Multimodal Mamba with Multitask Learning for Building Flood Damage Assessment using SAR Remote Sensing Imagery](https://arxiv.org/html/2506.06667)
37. [Multimodal Mamba with multitask learning for building flood damage assessment using synthetic aperture radar remote sensing imagery - Ho - 2025 - Computer-Aided Civil and Infrastructure Engineering - Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/mice.70059?af=R)
38. [Insta-BDA: Instance-Aware Building Damage Assessment and Counting via Foundation Model Fusion](https://www.mdpi.com/2072-4292/18/14/2347)
39. [\[2604.05959\] Multi-Modal Landslide Detection from Sentinel-1 SAR and Sentinel-2 Optical Imagery Using Multi-Encoder Vision Transformers and Ensemble Learning](https://arxiv.org/abs/2604.05959)
40. [\[2310.05959\] Automating global landslide detection with heterogeneous ensemble deep-learning classification](https://arxiv.org/abs/2310.05959)
41. [\[1911.09296\] xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://ar5iv.labs.arxiv.org/html/1911.09296)
42. [Global Flood Monitoring Product Launch](https://global-flood.emergency.copernicus.eu/news/107-global-flood-monitoring-product-launch/)
43. [Towards Digital Twin in Flood Forecasting with Data Assimilation Satellite Earth Observations -- A Proof-of-Concept in the Alzette Catchment](https://arxiv.org/pdf/2505.08553)
44. [Damage Proxy Maps and Flood Proxy Maps:](https://sentinel-asia.org/meetings/SA3JPTM7/agenda/Day2/Session4/4-14_EOS-JPL_SentinelAsia_2019_training2.pdf)
45. [A New Method for Large-Scale Landslide Classification from Satellite Radar](https://mdpi.com/2072-4292/11/3/237/htm)
46. [ARIA Damage Proxy Map (Copernicus Sentinel-1) for the Türkiye Earthquakes](https://esri-disasterresponse.hub.arcgis.com/maps/bd10c68dc2714258919c9bec4a4f3d02)
47. [Coherent and incoherent change detection for improved flood mapping: A Sentinel-1 SAR time-series approach - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352938526000947)
48. [Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002227)
49. [SAR-to-RGB Translation with Latent Diffusion for Earth Observation](https://arxiv.org/pdf/2504.11154)
50. [Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space MissionsThis research is partly supported by the National Science and Technology Council (NSTC), Taiwan, under Grant No. 113-2222-E-027-011 and 114-2119-M-011-001.The public source code of the proposed model is available at https://github.com/thoailt/Cloud-Removal.](https://arxiv.org/html/2506.17885v1)
51. [GLF-CR: SAR-Enhanced Cloud Removal with Global-Local Fusion](https://arxiv.org/pdf/2206.02850)
52. [GLF-CR: SAR-enhanced cloud removal with global–local fusion](https://www.researchgate.net/publication/363207116_GLF-CR_SAR-enhanced_cloud_removal_with_global-local_fusion)
53. [SAR-DeCR: Latent Diffusion for SAR-Fused Thick Cloud Removal](https://doi.org/10.3390/rs17132241)
54. [Multimodal Diffusion Bridge With Attention-Based SAR Fusion for Satellite Image Cloud Removal - WashU Research Profiles](https://profiles.wustl.edu/en/publications/multimodal-diffusion-bridge-with-attention-based-sar-fusion-for-s-2/)
55. [Cloud Removal in Remote Sensing Images Using Generative Adversarial Networks and SAR-to-Optical Image Translation](https://www.researchgate.net/publication/356551272_Cloud_Removal_in_Remote_Sensing_Images_Using_Generative_Adversarial_Networks_and_SAR-to-Optical_Image_Translation)
56. [\[2504.11171\] TerraMind: Large-Scale Generative Multimodality for Earth Observation](https://arxiv.org/abs/2504.11171)
57. [Clay-CNN Hybrids: Leveraging Geospatial Foundation Models as Auxiliary Context for Landslide Detection](https://arxiv.org/html/2606.14081)
58. [\[2206.00515\] Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection](https://ar5iv.labs.arxiv.org/html/2206.00515)
59. [Sentinel-1D Data Are Available for Download at ASF DAAC](https://www.earthdata.nasa.gov/data/alerts-outages/sentinel-1d-data-available-download-asf-daac)
60. [Sentinel-1D user data opening and future plans](https://dataspace.copernicus.eu/news/2026-4-2-sentinel-1d-user-data-opening-and-future-plans)
61. [OPERA Products](https://www.jpl.nasa.gov/go/opera/products/)
62. [▷ Discover OPERA DISP: Continental-Scale Deformations](https://detektia.com/en/opera-disp/)
63. [ARIA Damage Proxy Map Using ESA-Sentinel 1](https://appliedsciences.nasa.gov/our-impact/news/aria-damage-proxy-map-using-esa-sentinel-1)
64. [The fully-automatic Sentinel-1 Global Flood Monitoring service: Scientific challenges and future directions - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0034425725005127)
65. [Mapping Global Floods with 10 Years of Satellite Radar Data](https://arxiv.org/pdf/2411.01411)
66. [Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S003442572500286X)
67. [Urban Flood Observations: A global high-resolution](https://arxiv.org/pdf/2604.23066)
68. [Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 Bright Challenge](https://arxiv.org/html/2607.22746)
69. [NHESS - Pre-collapse motion of the February 2021 Chamoli rock–ice avalanche, Indian Himalaya](https://nhess.copernicus.org/articles/22/3309/2022/)
70. [(PDF) Pre-collapse motion of the February 2021 Chamoli rock-ice avalanche, Indian Himalaya](https://www.researchgate.net/publication/356114803_Pre-collapse_motion_of_the_February_2021_Chamoli_rock-ice_avalanche_Indian_Himalaya)
71. [Full article: DeepSAR Flood Mapper: global flood mapping on google earth engine cloud platform using MLP deep learning model with Sentinel-1 SAR imagery and HAND topographic data](https://www.tandfonline.com/doi/full/10.1080/15481603.2025.2612306)
72. [MSFlood-Net: A physically informed deep learning model integrating multi-source data for flood inundation mapping - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364815225004633)
73. [NASA/ISRO (Indian Space Research Organization) Synthetic Aperture Radar](https://www.earthdata.nasa.gov/data/platforms/space-based-platforms/nisar/data-access-tools)
74. [Suspected glacial lake outburst triggers catastrophic flood in Nepal](https://everestchronicle.com/nepal-glacial-lake-outburst-flood)
75. [Flooding in Nepal: What You Need to Know](https://internationalmedicalcorps.org/story/flooding-in-nepal-what-you-need-to-know/)
76. [OBSERVER: Countdown to launch](https://eu-space.europa.eu/news/observer-countdown-launch-copernicus-sentinel-1d-lifts-november)
