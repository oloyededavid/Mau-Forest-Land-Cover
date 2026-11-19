"""Printed maps and the chart for the paper and page, and the overlays for the interactive map.

Maps are drawn on the analysis grid (WGS 84 / UTM zone 36S), so the scale bar is exact, with a latitude and
longitude graticule, legend, scale bar, north arrow and credit line. Each map is written for desktop and phone.
Overlays for the interactive map are reprojected to Web Mercator and saved as PNG, with their bounds and legends
in docs/data/overlays.json.

    python scripts/s10_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import calculate_default_transform, reproject  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s07_compare  # noqa: E402
import study  # noqa: E402

MAPS = study.DOCS / "assets" / "maps"
OVER = study.DOCS / "data" / "overlays"
AUTHOR = "D. Oloyede"
DATE = "November 2026"
CRS_NOTE = "WGS 84 / UTM zone 36S (EPSG:32736)"
INK, INK2 = "#141412", "#55544e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.6})
AGREE = [(0, "Agrees with neither", "#b2182b"), (1, "Agrees with one", "#f4a582"), (2, "Agrees with both", "#e3e1da")]
LOSS_BINS = [((1, 8), "2001–2008", "#fdd49e"), ((9, 16), "2009–2016", "#fc8d59"), ((17, 24), "2017–2024", "#b30000")]
SIZES = {"desktop": (11.0, 8.2, 200), "phone": (5.6, 6.6, 150)}


def read(path, band=1, factor=4, resampling=Resampling.mode):
    with rasterio.open(path) as s:
        shape = (s.height // factor, s.width // factor)
        a = s.read(band, out_shape=shape, resampling=resampling)
        t = s.transform * s.transform.scale(s.width / shape[1], s.height / shape[0])
    return a, t


def extent(t, shape):
    return (t.c, t.c + shape[1] * t.a, t.f + shape[0] * t.e, t.f)


def box_outline():
    tr = Transformer.from_crs("EPSG:4326", study.CRS, always_xy=True)
    x0, y0, x1, y1 = study.BBOX
    lon = np.r_[np.linspace(x0, x1, 50), np.full(50, x1), np.linspace(x1, x0, 50), np.full(50, x0)]
    lat = np.r_[np.full(50, y0), np.linspace(y0, y1, 50), np.full(50, y1), np.linspace(y1, y0, 50)]
    return tr.transform(lon, lat)


def furniture(ax, ext, phone, credit):
    tr = Transformer.from_crs("EPSG:4326", study.CRS, always_xy=True)
    for lon in np.arange(35.4, 36.1, 0.2):
        xs, ys = tr.transform(np.full(60, lon), np.linspace(-1.0, -0.1, 60))
        ax.plot(xs, ys, color="white", lw=0.5, alpha=0.7, zorder=4)
        ax.text(xs[0], ext[2], f"{lon:.1f}°E", ha="center", va="top", fontsize=7, color=INK2)
    for lat in np.arange(-0.8, -0.1, 0.2):
        xs, ys = tr.transform(np.linspace(35.2, 36.2, 60), np.full(60, lat))
        ax.plot(xs, ys, color="white", lw=0.5, alpha=0.7, zorder=4)
        ax.text(ext[0], ys[0], f"{abs(lat):.1f}°S ", ha="right", va="center", fontsize=7, color=INK2)
    bx, by = box_outline()
    ax.plot(bx, by, color=INK, lw=0.9, zorder=5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    # scale bar 20 km, north arrow
    x0 = ext[0] + 0.04 * (ext[1] - ext[0]); y0 = ext[2] + 0.04 * (ext[3] - ext[2])
    for i in range(4):
        ax.add_patch(plt.Rectangle((x0 + i * 5000, y0), 5000, 900, facecolor=INK if i % 2 == 0 else "white",
                                   edgecolor=INK, lw=0.6, zorder=6))
    ax.add_patch(plt.Rectangle((x0 - 1500, y0 - 1500), 23500, 6500, facecolor="white", alpha=0.85,
                               edgecolor="none", zorder=5.5))
    ax.text(x0, y0 + 1600, "0", fontsize=7, ha="center", va="bottom", zorder=6)
    ax.text(x0 + 20000, y0 + 1600, "20 km", fontsize=7, ha="center", va="bottom", zorder=6)
    # north arrow outside the map frame, top right
    ax.annotate("N", xy=(1.035, 1.0), xytext=(1.035, 0.93), xycoords="axes fraction", ha="center", va="top",
                fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2),
                annotation_clip=False)
    import textwrap
    note = credit + f" {CRS_NOTE}. The map covers the study box, not the legal forest boundary. {AUTHOR}, {DATE}."
    ax.figure.text(0.02, 0.008, textwrap.fill(note, 95 if phone else 170), fontsize=6.2 if phone else 7,
                   color=INK2, va="bottom")


def save(fig, name):
    MAPS.mkdir(parents=True, exist_ok=True)
    tmp = MAPS / f"{name}.png"
    fig.savefig(tmp, dpi=fig.dpi, facecolor="white")
    plt.close(fig)
    Image.open(tmp).convert("RGB").save(MAPS / f"{name}.webp", "WEBP", quality=88, method=6)
    tmp.unlink()


def draw_map(name, title, subtitle, img, t, handles, credit):
    for kind, (w, h, dpi) in SIZES.items():
        phone = kind == "phone"
        if phone and not handles:   # no legend below the map: a shorter figure
            h = 5.7
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        if not phone:
            ax = fig.add_axes([0.07, 0.09, 0.62, 0.77])
        else:
            ax = fig.add_axes([0.11, 0.25, 0.85, 0.62] if handles else [0.11, 0.13, 0.85, 0.72])
        ext = extent(t, img.shape[:2])
        ax.imshow(img, extent=ext, interpolation="nearest", zorder=1)
        furniture(ax, ext, phone, credit)
        fig.text(0.02, 0.965, title, fontsize=13 if not phone else 11, fontweight="bold", color=INK, va="top")
        import textwrap
        fig.text(0.02, 0.925 if not phone else 0.935, textwrap.fill(subtitle, 120 if not phone else 70),
                 fontsize=9 if not phone else 7.5, color=INK2, va="top")
        if handles:
            if phone:
                fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.085), ncol=2, frameon=False,
                           fontsize=7.5)
            else:
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.72, 0.86), frameon=False, fontsize=9.5)
        save(fig, f"{name}_phone" if phone else name)


def paint(a, table):
    img = np.full(a.shape + (3,), 255, dtype=np.uint8)
    for code, hexcol in table.items():
        img[a == code] = (np.array(mcolors.to_rgb(hexcol)) * 255).astype(np.uint8)
    return img


def composite_rgb(factor=4):
    with rasterio.open(study.DATA / "composite.tif") as s:
        shape = (s.height // factor, s.width // factor)
        bands = [s.read(i, out_shape=shape, resampling=Resampling.average).astype(float) / 10000 for i in (3, 2, 1)]
        t = s.transform * s.transform.scale(s.width / shape[1], s.height / shape[0])
    rgb = np.clip(np.dstack(bands) / 0.16, 0, 1) ** 0.85
    return (rgb * 255).astype(np.uint8), t


def loss_classes(ly):
    out = np.zeros(ly.shape, dtype=np.uint8)
    for i, ((a, b), _, _) in enumerate(LOSS_BINS, start=1):
        out[(ly >= a) & (ly <= b)] = i
    return out


def overlay(name, arr, src_t, colours, out_w=1600):
    """Reproject a class array to Web Mercator and save an RGBA PNG; returns lat/lon bounds."""
    h, w = arr.shape
    left, top = src_t.c, src_t.f
    right, bottom = left + w * src_t.a, top + h * src_t.e
    dt, dw, dh = calculate_default_transform(study.CRS, "EPSG:3857", w, h, left, bottom, right, top)
    scale = out_w / dw
    dt = dt * dt.scale(1 / scale, 1 / scale)
    dw, dh = int(dw * scale), int(dh * scale)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(arr, dst, src_transform=src_t, src_crs=study.CRS, dst_transform=dt, dst_crs="EPSG:3857",
              resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    rgba = np.zeros((dh, dw, 4), dtype=np.uint8)
    for code, hexcol in colours.items():
        rgba[dst == code, :3] = (np.array(mcolors.to_rgb(hexcol)) * 255).astype(np.uint8)
        rgba[dst == code, 3] = 255
    OVER.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(OVER / f"{name}.png", optimize=True)
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = inv.transform(dt.c, dt.f + dh * dt.e)
    lon1, lat1 = inv.transform(dt.c + dw * dt.a, dt.f)
    return [[round(lat0, 6), round(lon0, 6)], [round(lat1, 6), round(lon1, 6)]]


def loss_chart(H):
    years = sorted(H["loss_km2_by_year"])
    vals = [H["loss_km2_by_year"][y] for y in years]
    for kind, (w, h, dpi) in {"desktop": (10, 4.4, 200), "phone": (5.6, 4.6, 150)}.items():
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.2)
        ax.bar([int(y) for y in years], vals, color="#b30000", width=0.75)
        ax.set_ylabel("Tree cover loss (km²)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([2001, 2005, 2010, 2015, 2020, 2024])
        fig.text(0.02, 0.96, "Hansen tree cover loss inside the study box, 2001–2024", fontsize=12 if kind == "desktop" else 9.5,
                 fontweight="bold", va="top")
        fig.text(0.02, 0.89, "Global Forest Change v1.12 (Hansen et al., 2013). A separate product; loss includes "
                 "plantation harvest.", fontsize=8.5 if kind == "desktop" else 6.8, color=INK2, va="top")
        save(fig, "m04_hansen_loss_years" + ("_phone" if kind == "phone" else ""))


def main():
    classes = json.loads((study.OUT / "classes.json").read_text())
    pal = {c["code"]: c["colour"] for c in classes["final"]}
    handles = [Patch(facecolor=c["colour"], edgecolor="none", label=c["name"]) for c in classes["final"]]
    scenes = json.loads((study.OUT / "composite.json").read_text())

    rgb, t = composite_rgb()
    draw_map("m01_composite", "Sentinel-2 dry-season composite, January–March 2025",
             f"Median of cloud-free observations from {scenes['n_scenes']} scenes, true colour (bands 4, 3, 2).",
             rgb, t, [], "Contains modified Copernicus Sentinel data 2025, via Microsoft Planetary Computer.")

    lc, t = read(study.DATA / "landcover_2025.tif")
    draw_map("m02_landcover", "Land cover, 2025 (this study)",
             "Random forest on the Sentinel-2 composite and terrain, trained on cells where two existing maps agree.",
             paint(lc, pal), t, handles,
             "Sentinel-2 L2A (Copernicus), Copernicus DEM GLO-30, labels from ESA WorldCover 2021 and Esri 2023.")

    ag, t = read(study.DATA / "agreement.tif")
    draw_map("m03_agreement", "Agreement with ESA WorldCover 2021 and Esri land cover 2023",
             "Each cell of the 2025 map against the two maps after translation into the same classes.",
             paint(ag, {k: c for k, _, c in AGREE}), t,
             [Patch(facecolor=c, edgecolor="#bbbbbb", lw=0.4, label=l) for _, l, c in AGREE],
             "ESA WorldCover 10 m 2021 v200 (Zanaga et al., 2022); Esri / Impact Observatory annual land cover 2023.")

    loss_chart(json.loads((study.OUT / "hansen_loss_by_year.json").read_text()))

    # web overlays (factor 2: 40 m cells)
    over = {"classes": classes["final"], "layers": []}
    lc2, t2 = read(study.DATA / "landcover_2025.tif", factor=2)
    over["layers"].append({"id": "this_map", "name": "This study, 2025", "file": "data/overlays/this_map.png",
                           "bounds": overlay("this_map", lc2, t2, pal),
                           "legend": [[c["name"], c["colour"]] for c in classes["final"]]})
    with rasterio.open(study.DATA / "layers.tif") as s:
        d = list(s.descriptions)
        shape = (s.height // 2, s.width // 2)
        L = {k: s.read(d.index(k) + 1, out_shape=shape, resampling=Resampling.nearest).astype(np.int16)
             for k in ("worldcover2021", "esri2023", "hansen_lossyear")}
    inbox = study.box_mask(t2, shape[1], shape[0])
    for key, label in (("worldcover2021", "ESA WorldCover 2021"), ("esri2023", "Esri land cover 2023")):
        a = s07_compare.translate(L[key], s07_compare.CROSSWALK[key])
        a[~inbox] = 0
        cols = dict(pal); cols[s07_compare.OTHER] = "#9a8f85"
        over["layers"].append({"id": key, "name": label, "file": f"data/overlays/{key}.png",
                               "bounds": overlay(key, a, t2, cols),
                               "legend": [[c["name"], c["colour"]] for c in classes["final"]] + [["Bare or other", "#9a8f85"]]})
    lo = loss_classes(L["hansen_lossyear"]); lo[~inbox] = 0
    over["layers"].append({"id": "hansen", "name": "Hansen tree cover loss 2001–2024", "file": "data/overlays/hansen.png",
                           "bounds": overlay("hansen", lo, t2, {i + 1: c for i, (_, _, c) in enumerate(LOSS_BINS)}),
                           "legend": [[l, c] for _, l, c in LOSS_BINS]})
    tr = Transformer.from_crs("EPSG:4326", "EPSG:4326")
    over["box"] = [[study.BBOX[1], study.BBOX[0]], [study.BBOX[3], study.BBOX[2]]]
    del tr
    (study.DOCS / "data" / "overlays.json").write_text(json.dumps(over, indent=1))
    print("maps:", sorted(p.name for p in MAPS.glob("*.webp")))


if __name__ == "__main__":
    main()
