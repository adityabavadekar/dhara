"""Offline gazetteer and location autocomplete for Dhara.

Provides fast, offline search for Himalayan districts, river corridors,
settlements, and landmark points without external network queries.
CRS for all geometries is EPSG:4326.
"""

from __future__ import annotations

from typing import Any

# Curated pre-event gazetteer for Nepal, the Trishuli-Bhotekoshi corridor,
# and key Himalayan benchmark sites. All bboxes are [minlon, minlat, maxlon, maxlat] in EPSG:4326.
PLACES: list[dict[str, Any]] = [
    {
        "id": "rasuwa_district",
        "name": "Rasuwa District",
        "name_ne": "रसुवा जिल्ला",
        "category": "district",
        "center": [85.35, 28.18],
        "bbox": [85.10, 27.95, 85.65, 28.40],
        "description": "Bagmati Province · Primary August 2026 flood & debris corridor",
    },
    {
        "id": "syabrubesi",
        "name": "Syabrubesi (Syapru Besi)",
        "name_ne": "स्याफ्रुबेसी",
        "category": "town",
        "center": [85.345, 28.155],
        "bbox": [85.315, 28.130, 85.375, 28.180],
        "description": "Major junction of Langtang Khola and Bhotekoshi river",
    },
    {
        "id": "dhunche",
        "name": "Dhunche",
        "name_ne": "धुन्चे",
        "category": "district_hq",
        "center": [85.300, 28.115],
        "bbox": [85.275, 28.090, 85.325, 28.140],
        "description": "Rasuwa District Headquarters & Hospital hub",
    },
    {
        "id": "timure",
        "name": "Timure",
        "name_ne": "टिमुरे",
        "category": "settlement",
        "center": [85.375, 28.275],
        "bbox": [85.350, 28.250, 85.400, 28.300],
        "description": "Customs port near Rasuwagadhi border checkpoint",
    },
    {
        "id": "rasuwagadhi",
        "name": "Rasuwagadhi Border",
        "name_ne": "रसुवागढी",
        "category": "border",
        "center": [85.385, 28.295],
        "bbox": [85.360, 28.270, 85.410, 28.320],
        "description": "Nepal–China border crossing on the Lhende / Bhotekoshi river",
    },
    {
        "id": "langtang_valley",
        "name": "Langtang Valley",
        "name_ne": "लाङटाङ उपत्यका",
        "category": "valley",
        "center": [85.500, 28.210],
        "bbox": [85.400, 28.150, 85.600, 28.270],
        "description": "Langtang National Park · Trail-dependent alpine communities",
    },
    {
        "id": "betrawati",
        "name": "Betrawati",
        "name_ne": "बेत्रावती",
        "category": "town",
        "center": [85.185, 27.975],
        "bbox": [85.155, 27.945, 28.005, 85.215],
        "description": "Trishuli river terrace · Historic border of Nuwakot/Rasuwa",
    },
    {
        "id": "bidur",
        "name": "Bidur / Battar",
        "name_ne": "विदुर / बट्टार",
        "category": "municipality",
        "center": [85.150, 27.910],
        "bbox": [85.110, 27.870, 85.190, 27.950],
        "description": "Nuwakot District Headquarters · Downstream floodplain",
    },
    {
        "id": "galchhi",
        "name": "Galchhi",
        "name_ne": "गल्छी",
        "category": "junction",
        "center": [85.015, 27.820],
        "bbox": [84.970, 27.780, 85.060, 27.860],
        "description": "Prithvi Highway & Trishuli confluence chokepoint",
    },
    {
        "id": "trishuli_corridor",
        "name": "Trishuli Full Impact Corridor",
        "name_ne": "त्रिशूली नदी बहाव क्षेत्र",
        "category": "corridor",
        "center": [85.250, 28.100],
        "bbox": [85.10, 27.85, 85.45, 28.35],
        "description": "Corridor from Rasuwagadhi down through Betrawati & Bidur (≈800 km²)",
    },
    {
        "id": "hakubeshi",
        "name": "Hakubeshi",
        "name_ne": "हाकुबेसी",
        "category": "village",
        "center": [85.265, 28.125],
        "bbox": [85.240, 28.100, 85.290, 28.150],
        "description": "Rasuwa trail village cut off by suspension bridge losses",
    },
    {
        "id": "gatlang",
        "name": "Gatlang",
        "name_ne": "गत्लाङ",
        "category": "village",
        "center": [85.260, 28.145],
        "bbox": [85.230, 28.120, 85.290, 28.170],
        "description": "Parvati Kunda highland settlement above Goljung",
    },
    {
        "id": "thambuchet",
        "name": "Thambuchet (Chilime)",
        "name_ne": "थाम्बुचेत",
        "category": "settlement",
        "center": [85.290, 28.190],
        "bbox": [85.260, 28.165, 85.320, 28.215],
        "description": "Chilime Hydropower hub in upper Rasuwa",
    },
    {
        "id": "devghat",
        "name": "Devghat",
        "name_ne": "देवघाट",
        "category": "confluence",
        "center": [84.425, 27.705],
        "bbox": [84.390, 27.670, 84.460, 27.740],
        "description": "Trishuli & Kali Gandaki confluence into the Narayani river",
    },
    {
        "id": "melamchi",
        "name": "Melamchi Valley",
        "name_ne": "मेलम्ची उपत्यका",
        "category": "valley",
        "center": [85.580, 27.830],
        "bbox": [85.500, 27.750, 85.660, 27.910],
        "description": "Sindhupalchok · Site of the June 2021 debris-flow disaster",
    },
    {
        "id": "chamoli",
        "name": "Chamoli / Rishiganga",
        "name_ne": "चमोली",
        "category": "valley",
        "center": [79.720, 30.480],
        "bbox": [79.600, 30.380, 79.840, 30.580],
        "description": "Uttarakhand, India · Benchmark Feb 2021 rock-ice avalanche",
    },
    {
        "id": "kathmandu",
        "name": "Kathmandu Valley",
        "name_ne": "काठमाडौं उपत्यका",
        "category": "city",
        "center": [85.320, 27.700],
        "bbox": [85.200, 27.600, 85.440, 27.800],
        "description": "National capital & emergency coordination center",
    },
]


def bbox_to_polygon(bbox: list[float]) -> dict[str, Any]:
    """Convert [minlon, minlat, maxlon, maxlat] to GeoJSON Polygon."""
    x0, y0, x1, y1 = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [round(x0, 4), round(y0, 4)],
                [round(x1, 4), round(y0, 4)],
                [round(x1, 4), round(y1, 4)],
                [round(x0, 4), round(y1, 4)],
                [round(x0, 4), round(y0, 4)],
            ]
        ],
    }


def search_places(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Search the gazetteer by English name, Nepali name, or category."""
    q = query.strip().lower()
    if not q:
        # Return prominent default places
        matches = PLACES[:limit]
    else:
        scored: list[tuple[int, dict[str, Any]]] = []
        for p in PLACES:
            name_en = p["name"].lower()
            name_ne = p.get("name_ne", "").lower()
            desc = p.get("description", "").lower()
            cat = p.get("category", "").lower()

            if q in name_en:
                score = 3 if name_en.startswith(q) else 2
            elif q in name_ne:
                score = 3
            elif q in desc or q in cat:
                score = 1
            else:
                score = 0

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [item[1] for item in scored[:limit]]

    results: list[dict[str, Any]] = []
    for m in matches:
        item = dict(m)
        item["polygon"] = bbox_to_polygon(item["bbox"])
        results.append(item)
    return results
