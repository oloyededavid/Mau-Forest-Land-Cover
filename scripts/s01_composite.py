"""Sentinel-2 L2A dry-season median composite over the study box, January to March 2025.

Scenes come from the Microsoft Planetary Computer STAC (collection sentinel-2-l2a, cloud cover below 20%).
Scenes from the same date and relative orbit are one acquisition; within an acquisition the first tile that has a
clear value fills a cell, so a date is never counted twice in tile overlaps. Cells are kept where the scene
classification (SCL) is vegetation (4), bare soil (5), water (6) or unclassified (7); cloud, cloud shadow,
cirrus, dark-area, saturated and snow cells are dropped. Reflectance = (DN - 1000) / 10000 for processing
baseline 04.00 and later (all scenes used here), stored as int16 x 10000.

    python scripts/s01_composite.py        # writes data/composite.tif, outputs/scenes.csv, outputs/composite.json

Memory: one band of every acquisition is held at once (about 2 GB for 40 acquisitions on the 20 m grid).
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
VALID_SCL = (4, 5, 6, 7)


def search():
    import planetary_computer as pc
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=study.BBOX, datetime=study.DATES,
                            query={"eo:cloud_cover": {"lt": study.MAX_CLOUD}}).items())
    items = [i for i in items if i.properties["s2:mgrs_tile"] in study.TILES]
    items.sort(key=lambda i: (i.datetime, i.properties["sat:relative_orbit"], i.properties["eo:cloud_cover"]))
    return items


def acquisitions(items):
    acq = defaultdict(list)
    for it in items:
        acq[(it.datetime.date().isoformat(), it.properties["sat:relative_orbit"], it.properties["platform"])].append(it)
    return dict(sorted(acq.items()))


def href(item, band):
    import planetary_computer as pc
    return pc.sign(item.assets[band].href)


def read_into(item, band, transform, width, height):
    """Read one band of one scene on the analysis grid. Returns (row slice, col slice, array) or None."""
    x0, y1 = transform.c, transform.f
    x1, y0 = x0 + width * study.RES, y1 - height * study.RES
    for attempt in range(4):
        try:
            with rasterio.open(href(item, band)) as src:
                assert src.crs.to_string() == study.CRS, src.crs
                b = src.bounds
                ix0, iy0, ix1, iy1 = max(b.left, x0), max(b.bottom, y0), min(b.right, x1), min(b.top, y1)
                if ix0 >= ix1 or iy0 >= iy1:
                    return None
                win = from_bounds(ix0, iy0, ix1, iy1, src.transform).round_offsets().round_lengths()
                h = int(round((iy1 - iy0) / study.RES)); w = int(round((ix1 - ix0) / study.RES))
                arr = src.read(1, window=win, out_shape=(h, w),
                               resampling=Resampling.nearest if band == "SCL" else Resampling.average)
            r0 = int(round((y1 - iy1) / study.RES)); c0 = int(round((ix0 - x0) / study.RES))
            return slice(r0, r0 + h), slice(c0, c0 + w), arr
        except Exception as e:  # network hiccups: retry with a fresh signature
            if attempt == 3:
                raise
            print("  retry", item.id, band, e, flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    study.DATA.mkdir(exist_ok=True); study.OUT.mkdir(exist_ok=True)
    transform, width, height = study.grid()
    items = search()
    acq = acquisitions(items)
    keys = list(acq)
    print(f"{len(items)} scenes, {len(keys)} acquisitions, grid {width} x {height}", flush=True)
    for it in items:
        bl = float(it.properties["s2:processing_baseline"])
        if bl < 4.0:
            raise SystemExit(f"{it.id} has baseline {bl}: the fixed -1000 offset would be wrong")

    # 1. clear-sky mask per acquisition (first tile with a clear value wins)
    n = len(keys)
    owner = np.full((n, height, width), -1, dtype=np.int16)   # index of the scene within the acquisition
    jobs = [(a, k, it) for a, key in enumerate(keys) for k, it in enumerate(acq[key])]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda j: (j, read_into(j[2], "SCL", transform, width, height)), jobs))
    for (a, k, it), res in results:
        if res is None:
            continue
        rs, cs, scl = res
        ok = np.isin(scl, VALID_SCL)
        sub = owner[a, rs, cs]
        sub[(sub < 0) & ok] = k
    clear = (owner >= 0).sum(axis=0).astype(np.uint8)
    print("clear observations per cell: median", int(np.median(clear)), "min", int(clear.min()), flush=True)

    # 2. median per band
    profile = dict(driver="GTiff", dtype="int16", width=width, height=height, count=len(study.S2_BANDS) + 1,
                   crs=study.CRS, transform=transform, nodata=-32768, compress="deflate", predictor=2,
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    out = study.DATA / "composite.tif"
    with rasterio.open(out, "w", **profile) as dst:
        for bi, band in enumerate(study.S2_BANDS, start=1):
            t0 = time.time()
            stack = np.zeros((n, height, width), dtype=np.uint16)
            with ThreadPoolExecutor(6) as ex:
                results = list(ex.map(lambda j: (j, read_into(j[2], band, transform, width, height)), jobs))
            for (a, k, it), res in results:
                if res is None:
                    continue
                rs, cs, arr = res
                sel = owner[a, rs, cs] == k
                stack[a, rs, cs][sel] = arr[sel]
                del arr
            del results
            med = np.full((height, width), -32768, dtype=np.int16)
            for r0 in range(0, height, 256):
                blk = stack[:, r0:r0 + 256].astype(np.float32)
                blk[owner[:, r0:r0 + 256] < 0] = np.nan
                blk[blk == 0] = np.nan
                with np.errstate(all="ignore"):
                    m = np.nanmedian(blk, axis=0)
                m = np.where(np.isfinite(m), np.clip(m - 1000.0, -32000, 32000), -32768)
                med[r0:r0 + 256] = np.round(m).astype(np.int16)
            dst.write(med, bi)
            dst.set_band_description(bi, band)
            del stack
            print(f"{band} done in {time.time() - t0:.0f} s", flush=True)
        dst.write(clear.astype(np.int16), len(study.S2_BANDS) + 1)
        dst.set_band_description(len(study.S2_BANDS) + 1, "clear_count")

    inbox = study.box_mask(transform, width, height)
    with open(study.OUT / "scenes.csv", "w") as f:
        f.write("scene_id,date,mgrs_tile,relative_orbit,platform,cloud_cover_pct,processing_baseline\n")
        for it in items:
            p = it.properties
            f.write(f"{it.id},{it.datetime.date()},{p['s2:mgrs_tile']},{p['sat:relative_orbit']},{p['platform']},"
                    f"{p['eo:cloud_cover']:.2f},{p['s2:processing_baseline']}\n")
    c = clear[inbox]
    meta = {
        "collection": "sentinel-2-l2a (Microsoft Planetary Computer)",
        "dates": study.DATES, "max_cloud_pct": study.MAX_CLOUD, "tiles": list(study.TILES),
        "n_scenes": len(items), "n_acquisitions": n,
        "first_date": str(items[0].datetime.date()), "last_date": str(items[-1].datetime.date()),
        "scenes_per_tile": {t: sum(1 for i in items if i.properties["s2:mgrs_tile"] == t) for t in study.TILES},
        "grid": {"crs": study.CRS, "res_m": study.RES, "width": width, "height": height,
                 "transform": list(transform)[:6]},
        "valid_scl": list(VALID_SCL),
        "clear_count_in_box": {"median": float(np.median(c)), "p05": float(np.percentile(c, 5)),
                               "min": int(c.min()), "max": int(c.max()),
                               "share_zero": float((c == 0).mean()), "share_below_3": float((c < 3).mean())},
    }
    (study.OUT / "composite.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
