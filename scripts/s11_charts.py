"""Figures drawn from the committed outputs only, so they can be rebuilt without the rasters in data/.

    m05_samples   where the training cells and the reference sample lie (locations only, no map class)
    m06_cv        agreement under random and spatially blocked cross-validation (cv.json)
    m07_areas     mapped area by class in this map, ESA WorldCover 2021 and Esri 2023 (comparison.json)

Each is written for desktop and phone in docs/assets/maps/, like the maps from s10_figures.py.

    python scripts/s11_charts.py
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pyproj import Transformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402
from s10_figures import AUTHOR, DATE, INK, INK2, SIZES, box_outline, furniture, save  # noqa: E402

FOLDS = [("random", "Random"), ("block_5km", "5 km blocks"), ("block_10km", "10 km blocks"), ("block_20km", "20 km blocks")]
NOTE_CV = "Agreement with labels taken from where two existing maps agree, on uniform cells. It is not map accuracy."


def heading(fig, title, subtitle, phone):
    if phone and len(title) > 36:
        title = textwrap.fill(title, 36, break_on_hyphens=False)
    top = fig.text(0.03, 0.975, title, fontsize=10.5 if phone else 13, fontweight="bold", color=INK, va="top")
    fig.canvas.draw()
    y = fig.transFigure.inverted().transform(top.get_window_extent())[0][1] - 0.012
    fig.text(0.03, y if phone else 0.925, textwrap.fill(subtitle, 56 if phone else 120),
             fontsize=7.5 if phone else 9, color=INK2, va="top")


def credit(fig, text, phone):
    fig.text(0.03, 0.008, textwrap.fill(f"{text} {AUTHOR}, {DATE}.", 68 if phone else 170),
             fontsize=6.5 if phone else 7, color=INK2, va="bottom")


def samples():
    """Training cells by label class, and the 644 reference points (locations only)."""
    tr = pd.read_csv(study.OUT / "training_samples.csv", usecols=["x", "y", "label"])
    pts = json.loads((study.DOCS / "data" / "label_points.json").read_text())["points"]
    to_utm = Transformer.from_crs("EPSG:4326", study.CRS, always_xy=True)
    px, py = to_utm.transform([p["lon"] for p in pts], [p["lat"] for p in pts])
    bx, by = box_outline()
    pad = 2500
    ext = (min(bx) - pad, max(bx) + pad, min(by) - pad, max(by) + pad)
    cls = study.CANDIDATE_CLASSES
    for kind, (w, h, dpi) in SIZES.items():
        phone = kind == "phone"
        if phone:
            w, dpi = 3.9, 215
        fig = plt.figure(figsize=(w, 7.4 if phone else 6.4), dpi=dpi)
        if phone:
            axes = [fig.add_axes([0.14, 0.47, 0.76, 0.3]), fig.add_axes([0.14, 0.13, 0.76, 0.3])]
        else:
            axes = [fig.add_axes([0.06, 0.12, 0.42, 0.68]), fig.add_axes([0.55, 0.12, 0.42, 0.68])]
        a0, a1 = axes
        for code in sorted(tr["label"].unique()):
            d = tr[tr["label"] == code]
            a0.scatter(d["x"], d["y"], s=0.6 if not phone else 0.35, color=cls[code]["colour"], lw=0, zorder=3,
                       rasterized=True)
        a1.scatter(px, py, s=7 if not phone else 4, facecolor="white", edgecolor=INK, lw=0.6, zorder=3)
        for ax, t in ((a0, f"Training cells ({len(tr):,})"), (a1, f"Reference sample ({len(pts)} points)")):
            ax.set_facecolor("#f3f2ee")
            furniture(ax, ext, phone, "")
            ax.set_title(t, fontsize=9.5 if not phone else 8.5, loc="left", color=INK)
        # furniture writes a credit line per axis; replace both with one
        for t in list(fig.texts):
            if "study box, not the legal" in t.get_text():
                t.remove()
        handles = [Line2D([], [], marker="o", ls="", color=cls[c]["colour"], label=cls[c]["name"], markersize=5)
                   for c in sorted(tr["label"].unique())]
        handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK,
                              label="Reference point (no class shown)", markersize=5))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.045 if phone else 0.035),
                   ncol=2 if phone else 6, frameon=False, fontsize=6.8 if phone else 8)
        heading(fig, "Where the training cells and the reference sample lie",
                "Training cells: where ESA WorldCover 2021 and Esri 2023 agree in a uniform neighbourhood, up to 4,000 "
                "per class, before plantation was merged into forest. Reference points: the stratified sample for the "
                "accuracy assessment, shown without their map class.", phone)
        credit(fig, "Map covers the study box, not the legal forest boundary. WGS 84 / UTM zone 36S (EPSG:32736).", phone)
        save(fig, "m05_samples" + ("_phone" if phone else ""))


def cv_chart():
    cv = json.loads((study.OUT / "cv.json").read_text())
    series = [("final_scheme", "random_forest", "Random forest, final classes", INK, "-", "o"),
              ("final_scheme", "gradient_boosting", "Gradient boosting, final classes", "#4b6e9c", "--", "s"),
              ("candidate_scheme", "random_forest", "Random forest, plantation kept", "#b3361b", "-", "^")]
    for kind, (w, h, dpi) in {"desktop": (10, 4.8, 200), "phone": (3.9, 4.6, 215)}.items():
        phone = kind == "phone"
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        fig.subplots_adjust(left=0.16 if phone else 0.08, right=0.95, top=0.78 if not phone else 0.7,
                            bottom=0.33 if phone else 0.24)
        x = np.arange(len(FOLDS))
        lows = [cv[sc]["results"][k]["models"][m]["overall_accuracy"] for sc, m, *_ in series for k, _ in FOLDS]
        for scheme, model, label, colour, ls, mk in series:
            y = [cv[scheme]["results"][k]["models"][model]["overall_accuracy"] for k, _ in FOLDS]
            ax.plot(x, y, ls=ls, marker=mk, color=colour, lw=1.6, ms=5, label=label)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 7 if mk == "s" else -12), textcoords="offset points",
                            ha="center", fontsize=6.5 if phone else 7.5, color=colour)
        ax.set_xticks(x, [l.replace(' blocks', '') if phone else l for _, l in FOLDS], fontsize=8 if phone else 9)
        ax.set_ylabel("Overall agreement")
        ax.set_ylim(np.floor(min(lows) * 50) / 50 - 0.01, 1.0)
        ax.grid(axis="y", color="#e5e5e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12 if not phone else -0.14), ncol=1 if phone else 3,
                  frameon=False, fontsize=7 if phone else 8.5)
        heading(fig, "Cross-validated agreement, random against spatially blocked folds",
                f"Five folds, {cv['n_samples']:,} training cells. Blocks hold out whole areas of 5, 10 or 20 km. " + NOTE_CV,
                phone)
        credit(fig, "Source: outputs/cv.json; the y-axis does not start at zero.", phone)
        save(fig, "m06_cv" + ("_phone" if phone else ""))


def areas_chart():
    c = json.loads((study.OUT / "comparison.json").read_text())["areas_km2"]
    classes = [k for k in c if not k.startswith("Other")]
    products = [("worldcover2021", "ESA WorldCover 2021", "#b9b8b1"), ("this_map", "This map, 2025", INK),
                ("esri2023", "Esri land cover 2023", "#7d7c76")]
    for kind, (w, h, dpi) in {"desktop": (10, 4.8, 200), "phone": (3.9, 5.4, 215)}.items():
        phone = kind == "phone"
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        fig.subplots_adjust(left=0.3 if phone else 0.22, right=0.93, top=0.8 if not phone else 0.72,
                            bottom=0.14 if phone else 0.16)
        y = np.arange(len(classes))
        hgt = 0.26
        for i, (key, label, colour) in enumerate(products):
            v = [c[k][key] for k in classes]
            ax.barh(y + (i - 1) * hgt, v, height=hgt * 0.92, color=colour, label=label)
            for yy, vv in zip(y + (i - 1) * hgt, v):
                ax.text(vv + 40, yy, f"{vv:,.0f}", va="center", fontsize=6.5 if phone else 7.5, color=INK2)
        ax.set_yticks(y, [textwrap.fill(k, 18) for k in classes], fontsize=7.5 if phone else 9)
        ax.invert_yaxis()
        ax.set_xlabel("Area inside the study box (km²)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e5e1", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5 if phone else 8.5)
        heading(fig, "Mapped area by class in three land-cover maps",
                "Pixel counts after translating each map into the same classes; not area estimates with confidence "
                "intervals, which wait for the reference sample.", phone)
        credit(fig, "ESA WorldCover 2021, Esri / Impact Observatory 2023 and this map; source: outputs/comparison.json.", phone)
        save(fig, "m07_areas" + ("_phone" if phone else ""))


def main():
    samples()
    cv_chart()
    areas_chart()
    print("charts: m05_samples, m06_cv, m07_areas")


if __name__ == "__main__":
    main()
