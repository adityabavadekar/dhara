# Architecture

Dhara turns satellite images taken before and after an event into three answers: where the flood
hit, what was damaged, and who is cut off.

```mermaid
flowchart LR
    subgraph Inputs
        S1[Sentinel-1 radar]
        S2[Sentinel-2 optical]
        DEM[Copernicus DEM]
        OSM[OpenStreetMap before the event]
    end

    S1 --> SAR[Radar processing<br/>calibrate, filter, geocode]
    DEM --> SAR
    DEM --> TER[Terrain<br/>height above river, slope]

    SAR --> DET[Detectors<br/>radar change, U-Net]
    S2 --> DET
    DET --> FUSE[Fusion<br/>per-pixel class and confidence]
    TER --> FUSE
    FUSE --> PHY[Physics rules<br/>keep only what a flood can do]
    TER --> PHY

    PHY --> DMG[Damage<br/>buildings, roads, bridges]
    OSM --> DMG
    DMG --> NET[Road network<br/>reroute every settlement]
    OSM --> NET

    NET --> OUT[Dashboard, assistant,<br/>one-page report]
    DMG --> OUT
```

## Stages

| Stage | Code | What it does |
|---|---|---|
| Radar processing | `src/dhara/sar/` | Reads raw Sentinel-1 GRD files: noise removal, calibration, speckle filter, terrain geocoding, terrain flattening, layover and shadow masks |
| Terrain | `src/dhara/terrain/` | Height above the nearest river and slope, from the Copernicus DEM |
| Detectors | `src/dhara/members/` | Radar water and change, optical water and debris, and the flood U-Net |
| Fusion | `src/dhara/fusion/fuse.py` | Combines the detectors into flood, debris, no change or not observable, with a confidence level |
| Physics rules | `src/dhara/fusion/detector.py` | Removes water too high above the river or on steep slopes, keeps debris connected to the river, fills the valley floor up to the mapped flood level |
| Damage | `src/dhara/damage/` | Marks buildings, roads and bridges inside the footprint |
| Road network | `src/dhara/network/` | Routes each settlement to a town and a hospital before and after the event, and ranks repairs |
| Outputs | `src/dhara/api/`, `src/dhara/sitrep/`, `src/dhara/assistant/`, `web/` | Dashboard, grounded assistant and one-page report in English and Nepali |

## Flood model

```mermaid
flowchart LR
    IN[6 radar bands<br/>VV, VH: after, before, before] --> ENC[ResNet-34 encoder<br/>ImageNet weights]
    ENC --> DEC[U-Net decoder]
    DEC --> OUT[3 classes<br/>no water, permanent water, flood]
```

- Trained on Kuro Siwo, with mountain events held out by terrain relief.
- Exported to ONNX and run on CPU.
- Weights: [huggingface.co/adityabavadekar/dhara-flood-unet](https://huggingface.co/adityabavadekar/dhara-flood-unet).
