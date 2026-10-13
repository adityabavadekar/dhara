"""Exploratory Data Analysis (EDA) and Soft Validation: HDX / UN Nepal Flash Flood 2026.

NOTE: This script lives exclusively in validation/ for offline research, soft validation,
and case-study evaluation. It is strictly excluded from src/dhara/ and production pipelines
to guarantee zero compliance violations.

Compares:
  - HDX / UNOSAT / IHME 2026 Nepal Flash Flood population impact extent
  - Dhara Multi-Hazard Inundation & Debris footprint from job bd970069311d (Rasuwa District)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def run_eda():
    val_dir = Path("validation")
    plots_dir = val_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hdx_tif = val_dir / "data" / "nepal_flash_flood_2026_ihme_population_extent.tif"
    dhara_job_dir = Path("data/jobs/bd970069311d")

    if not hdx_tif.exists() or not dhara_job_dir.exists():
        print(f"Error: Missing {hdx_tif} or {dhara_job_dir}")
        return

    print("Loading HDX Population Extent Raster...")
    with rasterio.open(hdx_tif) as src_hdx:
        src_hdx.meta.copy()
        hdx_crs = src_hdx.crs

        # Load Dhara's p_debris and p_flood
        print("Reprojecting Dhara predictions to HDX reference grid...")
        with rasterio.open(dhara_job_dir / "p_debris.tif") as src_deb:
            dhara_deb_reproj = np.zeros(src_hdx.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src_deb, 1),
                destination=dhara_deb_reproj,
                src_transform=src_deb.transform,
                src_crs=src_deb.crs,
                dst_transform=src_hdx.transform,
                dst_crs=hdx_crs,
                resampling=Resampling.bilinear,
            )

        with rasterio.open(dhara_job_dir / "p_flood.tif") as src_fld:
            dhara_fld_reproj = np.zeros(src_hdx.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src_fld, 1),
                destination=dhara_fld_reproj,
                src_transform=src_fld.transform,
                src_crs=src_fld.crs,
                dst_transform=src_hdx.transform,
                dst_crs=hdx_crs,
                resampling=Resampling.bilinear,
            )

        hdx_pop = src_hdx.read(1)

    # Combined Dhara hazard mask (> 0.4 probability)
    dhara_hazard = (dhara_deb_reproj >= 0.4) | (dhara_fld_reproj >= 0.4)
    # Valid overlap zone where Dhara analyzed
    dhara_aoi = np.isfinite(dhara_deb_reproj) & (dhara_deb_reproj > 0)

    # Compute crop window focusing on Dhara's AOI in the HDX raster
    rows, cols = np.where(dhara_aoi)
    if len(rows) == 0:
        print("Warning: No overlapping pixels found between Dhara AOI and HDX raster.")
        return

    r_min, r_max = max(0, rows.min() - 30), min(hdx_pop.shape[0], rows.max() + 30)
    c_min, c_max = max(0, cols.min() - 30), min(hdx_pop.shape[1], cols.max() + 30)

    crop_hdx_pop = hdx_pop[r_min:r_max, c_min:c_max]
    crop_dhara_deb = dhara_deb_reproj[r_min:r_max, c_min:c_max]
    crop_dhara_fld = dhara_fld_reproj[r_min:r_max, c_min:c_max]
    crop_hazard = dhara_hazard[r_min:r_max, c_min:c_max]

    # Compute Soft Statistical Metrics
    pop_in_aoi = np.nansum(crop_hdx_pop[crop_hdx_pop > 0])
    pop_exposed_dhara = np.nansum(crop_hdx_pop[crop_hazard & (crop_hdx_pop > 0)])
    area_debris_km2 = float((crop_dhara_deb >= 0.5).sum() * (src_hdx.res[0] * src_hdx.res[1]) / 1e6)
    area_flood_km2 = float((crop_dhara_fld >= 0.5).sum() * (src_hdx.res[0] * src_hdx.res[1]) / 1e6)

    print("\n" + "=" * 70)
    print("SOFT VALIDATION & EDA SUMMARY: HDX NEPAL FLASH FLOOD VS DHARA")
    print("=" * 70)
    print(f"Total Regional Population in Watershed (IHME): {pop_in_aoi:,.1f}")
    print(f"Estimated Population Exposed to Dhara Hazard:   {pop_exposed_dhara:,.1f}")
    print(f"Dhara Detected Debris Footprint Area:          {area_debris_km2:.2f} km²")
    print(f"Dhara Detected Open Flood Area:                {area_flood_km2:.2f} km²")
    print("Official UNOSAT Reported Impact Footprint:     37.0 km²")
    print("Official UN HDX Intersecting Overture Bldgs:   4,977")
    print("=" * 70)

    # Plot Multi-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    # 1. HDX IHME Population Density Map
    im1 = axes[0, 0].imshow(np.log1p(np.clip(crop_hdx_pop, 0, None)), cmap="magma")
    axes[0, 0].set_title(
        "1. HDX / IHME 2026 Population Density (log scale)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].axis("off")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label="log(1 + Population)")

    # 2. Dhara Multi-Hazard Prediction (Debris in Orange, Water in Blue)
    hazard_rgb = np.zeros((*crop_dhara_deb.shape, 3), dtype=np.float32)
    # Background terrain gray
    hazard_rgb[:] = [0.88, 0.88, 0.88]
    # Debris in orange
    is_deb = crop_dhara_deb >= 0.4
    hazard_rgb[is_deb] = [0.85, 0.45, 0.05]
    # Flood in blue
    is_fld = crop_dhara_fld >= 0.4
    hazard_rgb[is_fld] = [0.1, 0.4, 0.85]

    axes[0, 1].imshow(hazard_rgb)
    axes[0, 1].set_title(
        "2. Dhara Detected Footprint (Orange: Debris, Blue: Flood)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].axis("off")

    # 3. Direct Spatial Overlay
    overlay = np.log1p(np.clip(crop_hdx_pop, 0, None))
    axes[1, 0].imshow(overlay, cmap="bone")
    # Draw hazard contour on top
    axes[1, 0].contour(crop_hazard, levels=[0.5], colors=["#ef4444"], linewidths=1.5)
    axes[1, 0].set_title(
        "3. Overlay: Population Density + Dhara Impact Perimeter (Red)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    # 4. Statistical Comparison Bar Chart
    categories = [
        "UNOSAT Total Area (km²)",
        "Dhara Hazard Area (km²)",
        "HDX Impacted Pop (×1k)",
        "Dhara Exposed Pop (×1k)",
    ]
    vals = [37.0, area_debris_km2 + area_flood_km2, 10.2, pop_exposed_dhara / 1000.0]
    colors = ["#64748b", "#2563eb", "#64748b", "#10b981"]

    bars = axes[1, 1].bar(categories, vals, color=colors, edgecolor="#0f172a", width=0.55)
    axes[1, 1].set_title("4. Ground Truth vs Model Soft Alignment", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Quantity", fontsize=11)
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[1, 1].annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Exploratory Validation: UN HDX / UNOSAT Reference Map vs Dhara Zero-Shot Prediction",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    out_img = plots_dir / "hdx_vs_dhara_nepal_eda.png"
    fig.savefig(out_img, dpi=200)
    plt.close(fig)
    print(f"Saved EDA and soft validation visual to {out_img}")


if __name__ == "__main__":
    run_eda()
