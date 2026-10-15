"""Charts for validation/results.md, drawn from the numbers reported there (no new scoring).

    uv run python validation/plot_results.py

Writes PNGs to validation/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, AMBER, GREY, GREEN = "#2563eb", "#d97706", "#9ca3af", "#16a34a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 34,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    }
)


def _label(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def detector_gain():
    """Section 1a: EMS F1, fusion only vs physics detector."""
    areas = ["01 Syapru Besi", "02 Timure", "03 Bidur", "05 Phosretar\n(held out)"]
    fusion = [0.67, 0.50, 0.73, 0.53]
    detector = [0.85, 0.79, 0.88, 0.84]
    x = range(len(areas))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _label(ax, ax.bar([i - w / 2 for i in x], fusion, w, color=GREY, label="Fusion only"))
    bars = ax.bar([i + w / 2 for i in x], detector, w, color=BLUE, label="With physics detector")
    bars[-1].set_color(AMBER)
    _label(ax, bars)
    ax.set_xticks(list(x), areas)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 vs Copernicus EMS")
    ax.set_title("Footprint F1 against EMSR927")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    ax.text(3.19, 0.97, "held out, scored once", ha="center", fontsize=9, color=AMBER)
    _save(fig, "detector_gain.png")


def controls():
    """Section 1a: radar only, flood run vs pre-flood control, with the detector."""
    areas = ["01 Syapru Besi", "02 Timure"]
    flood = [0.48, 0.46]
    control = [0.00, 0.01]
    x = range(len(areas))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.8))
    _label(ax, ax.bar([i - w / 2 for i in x], flood, w, color=BLUE, label="Flood run"))
    _label(
        ax,
        ax.bar(
            [i + w / 2 for i in x],
            control,
            w,
            color=GREY,
            label="Control, all images before 26 Aug",
        ),
    )
    ax.set_xticks(list(x), areas)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("F1 vs EMS, radar only")
    ax.set_title("No flood, no footprint")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    _save(fig, "controls.png")


def model_scores():
    """Section 6: flood F1 on Kuro Siwo test events and the unseen mountain fold."""
    names = ["v1 scratch\n(retired)", "v2\n(deployed)", "v3", "v4", "v6", "v7"]
    test = [0.80, 0.74, 0.74, 0.73, 0.72, 0.72]
    test_ci = [(0.79, 0.86), (0.62, 0.85), (0.66, 0.85), (0.60, 0.85), None, None]
    mountain = [0.73, 0.75, 0.76, 0.75, 0.76, 0.76]
    mountain_ci = [None, (0.69, 0.84), (0.71, 0.84), (0.71, 0.83), None, None]

    def err(vals, cis):
        lo = [v - c[0] if c else 0 for v, c in zip(vals, cis, strict=True)]
        hi = [c[1] - v if c else 0 for v, c in zip(vals, cis, strict=True)]
        return [lo, hi]

    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    kw = {"capsize": 3, "error_kw": {"ecolor": MUTED, "elinewidth": 1}}
    b1 = ax.bar(
        [i - w / 2 for i in x],
        test,
        w,
        color=GREY,
        label="Test events (7)",
        yerr=err(test, test_ci),
        **kw,
    )
    b2 = ax.bar(
        [i + w / 2 for i in x],
        mountain,
        w,
        color=BLUE,
        label="Unseen mountain events (4)",
        yerr=err(mountain, mountain_ci),
        **kw,
    )
    b1[0].set_hatch("//")
    b1[0].set_edgecolor("white")
    b2[1].set_color(AMBER)
    for b in (*b1, *b2):
        ax.annotate(
            f"{b.get_height():.2f}",
            (b.get_x() + b.get_width() / 2, 0.03),
            ha="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Flood F1 (Kuro Siwo)")
    ax.set_title("Flood segmentation models on the same held-out tiles")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    ax.text(-0.4, 0.95, "v1: validation and test were the same tiles", fontsize=8.5, color=MUTED)
    _save(fig, "model_scores.png")


def sen1floods11():
    """Section 6: Sen1Floods11 water IoU, test split and the unseen Bolivia event."""
    names = ["Otsu on VV\n(no model)", "v1 scratch", "v2\n(deployed)", "v3", "v4", "v6*"]
    test = [0.21, 0.29, 0.61, 0.48, 0.55, 0.65]
    bolivia = [0.35, 0.63, 0.71, 0.72, 0.72, 0.57]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], test, w, color=GREY, label="Sen1Floods11 test")
    b2 = ax.bar([i + w / 2 for i in x], bolivia, w, color=BLUE, label="Bolivia (unseen country)")
    b1[2].set_color(AMBER)
    b2[2].set_color(AMBER)
    _label(ax, b1)
    _label(ax, b2)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Water IoU")
    ax.set_title("Sen1Floods11")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    ax.text(5.4, 0.9, "*trained on\nSen1Floods11", ha="right", fontsize=8.5, color=MUTED)
    _save(fig, "sen1floods11.png")


def model_vs_system():
    """Sections 1a and 6b: the water model alone vs the whole system on EMSR927."""
    areas = ["01 Syapru Besi", "02 Timure", "03 Bidur"]
    model = [0.00, 0.00, 0.04]
    system = [0.85, 0.79, 0.88]
    x = range(len(areas))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    _label(ax, ax.bar([i - w / 2 for i in x], model, w, color=GREY, label="Flood model alone (v2)"))
    _label(ax, ax.bar([i + w / 2 for i in x], system, w, color=BLUE, label="Whole system"))
    ax.set_xticks(list(x), areas)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 vs EMS")
    ax.set_title("A water model alone misses debris; fusion and physics find it")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    _save(fig, "model_vs_system.png")


def cutoff():
    """Section 1c: settlements with the same access status as EMS-based routing."""
    areas = ["01 Syapru Besi\n127", "02 Timure\n115", "03 Bidur\n1,383", "05 Phosretar\n1,534"]
    same = [100, 99, 89, 80]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(areas, same, 0.55, color=[GREEN, GREEN, BLUE, BLUE])
    _label(ax, bars, "{:.0f} %")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Same status as EMS routing (%)")
    ax.set_xlabel("EMS area and settlements checked", color=MUTED)
    ax.set_title("Who is cut off: agreement with EMS-damaged roads")
    _save(fig, "cutoff_agreement.png")


def s1_chain():
    """Section 1b: whole-system F1 with our Sentinel-1 chain vs Sentinel Hub."""
    areas = ["01 Syapru Besi", "02 Timure", "05 Phosretar"]
    ours = [0.85, 0.80, 0.84]
    hub = [0.85, 0.79, 0.84]
    x = range(len(areas))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _label(ax, ax.bar([i - w / 2 for i in x], ours, w, color=BLUE, label="Own chain, raw GRD"))
    _label(ax, ax.bar([i + w / 2 for i in x], hub, w, color=GREY, label="Sentinel Hub"))
    ax.set_xticks(list(x), areas)
    ax.set_ylim(0, 1)
    ax.set_ylabel("System F1 vs EMS")
    ax.set_title("Own Sentinel-1 processing matches Sentinel Hub")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=10, borderaxespad=0.2)
    _save(fig, "s1_chain.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (
        detector_gain,
        controls,
        model_scores,
        sen1floods11,
        model_vs_system,
        cutoff,
        s1_chain,
    ):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
