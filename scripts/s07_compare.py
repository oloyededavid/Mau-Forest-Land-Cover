"""Compare the 2025 map with ESA WorldCover 2021, Esri land cover 2023 and Hansen forest loss, and with the 21
Lesiv et al. (2025) reference sites from 2015 (a descriptive check only).

WorldCover and Esri classes are translated into the study's classes (CROSSWALK). Classes the study does not map
(bare ground, clouds) become "Other".

    python scripts/s07_compare.py    # writes outputs/comparison.json, data/agreement.tif
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

OTHER = 9
CROSSWALK = {
    "worldcover2021": {10: 1, 20: 3, 30: 3, 40: 3, 50: 4, 60: OTHER, 80: 5, 90: 5, 95: 5, 100: OTHER},
    "esri2023": {2: 1, 5: 3, 11: 3, 7: 4, 1: 5, 4: 5, 8: OTHER, 9: OTHER, 10: OTHER},
}
LESIV = {"tree": 1, "crops": 3, "grassland": 3, "shrub": 3, "fallow/shifting cultivation": 3,
         "urban/built-up": 4, "water": 5, "wetland": 5, "bare": OTHER}


def translate(a, table):
    out = np.zeros(a.shape, dtype=np.uint8)
    for k, v in table.items():
        out[a == k] = v
    return out


def area_table(ours, other, codes, names, cell_km2):
    """km2 by (our class, other class)."""
    t = {}
    for a in codes:
        t[names[a]] = {names[b]: float(((ours == a) & (other == b)).sum() * cell_km2) for b in codes + [OTHER]}
    return t


def main():
    classes = json.loads((study.OUT / "classes.json").read_text())
    codes = [c["code"] for c in classes["final"]]
    names = {c["code"]: c["name"] for c in classes["final"]}
    names[OTHER] = "Other (bare, cloud)"
    transform, width, height = study.grid()
    inbox = study.box_mask(transform, width, height)
    with rasterio.open(study.DATA / "landcover_2025.tif") as s:
        ours = s.read(1)
    with rasterio.open(study.DATA / "layers.tif") as s:
        d = list(s.descriptions)
        L = {k: s.read(d.index(k) + 1).astype(np.int16) for k in
             ("worldcover2021", "esri2023", "hansen_lossyear", "hansen_treecover2000", "hansen_datamask")}
    valid = inbox & (ours > 0)
    cell = study.RES ** 2 / 1e6
    wc = translate(L["worldcover2021"], CROSSWALK["worldcover2021"])
    es = translate(L["esri2023"], CROSSWALK["esri2023"])
    valid &= (wc > 0) & (es > 0)
    o, w, e = ours[valid], wc[valid], es[valid]

    agree_n = (o == w).astype(np.uint8) + (o == e).astype(np.uint8)
    agreement = np.full(ours.shape, 255, dtype=np.uint8)
    agreement[valid] = agree_n
    prof = dict(driver="GTiff", dtype="uint8", width=width, height=height, count=1, crs=study.CRS,
                transform=transform, nodata=255, compress="deflate", tiled=True)
    with rasterio.open(study.DATA / "agreement.tif", "w", **prof) as dst:
        dst.write(agreement, 1)

    areas = {names[k]: {"this_map": float((o == k).sum() * cell), "worldcover2021": float((w == k).sum() * cell),
                        "esri2023": float((e == k).sum() * cell)} for k in codes + [OTHER]}
    per_class = {}
    for k in codes:
        m = o == k
        per_class[names[k]] = {"share_agreeing_with_worldcover": float((w[m] == k).mean()),
                               "share_agreeing_with_esri": float((e[m] == k).mean()),
                               "share_agreeing_with_both": float(((w[m] == k) & (e[m] == k)).mean())}
    # largest disagreements: (ours, worldcover, esri) triples by area
    trip = pd.Series(list(zip(o[agree_n < 2], w[agree_n < 2], e[agree_n < 2]))).value_counts().head(8)
    top = [{"this_map": names[a], "worldcover2021": names[b], "esri2023": names[c], "area_km2": float(n * cell)}
           for (a, b, c), n in trip.items()]
    # where the two existing maps disagree with each other, which one does this map side with?
    dis = w != e
    side = {"cells_km2": float(dis.sum() * cell), "with_worldcover": float((o[dis] == w[dis]).mean()),
            "with_esri": float((o[dis] == e[dis]).mean()),
            "with_neither": float(((o[dis] != w[dis]) & (o[dis] != e[dis])).mean())}

    # Hansen: context only, a separate product
    ly = L["hansen_lossyear"][valid]
    forest_code = [c["code"] for c in classes["final"] if c["name"].startswith("Forest")][0]
    lost = ly > 0
    hansen = {
        "loss_cells_km2_on_grid": float(lost.sum() * cell),
        "share_of_loss_2001_2024_mapped_forest_2025": float((o[lost] == forest_code).mean()),
        "share_of_loss_2021_2024_mapped_forest_2025": float((o[ly >= 21] == forest_code).mean()),
        "loss_2001_2024_by_class_km2": {names[k]: float(((o == k) & lost).sum() * cell) for k in codes},
        "treecover2000_ge30_km2": float((L["hansen_treecover2000"][valid] >= 30).sum() * cell),
        "treecover2000_ge30_mapped_forest_2025_km2": float(((L["hansen_treecover2000"][valid] >= 30)
                                                           & (o == forest_code)).sum() * cell),
        "share_mapped_forest_with_treecover2000_ge30": float((L["hansen_treecover2000"][valid][o == forest_code]
                                                              >= 30).mean()),
    }

    # Lesiv 2015 reference sites: dominant class of each 100 m square against this map's 5 x 5 cells (20 m)
    ref = pd.read_csv(study.OUT / "reference_lesiv2015_box.csv")
    ref["cls"] = ref["class_name"].map(LESIV).fillna(OTHER).astype(int)
    tr = Transformer.from_crs("EPSG:4326", study.CRS, always_xy=True)
    sites = []
    for sid, g in ref.groupby("sampleid"):
        x, y = tr.transform(g["center_x"].mean(), g["center_y"].mean())
        col = int((x - transform.c) / study.RES); row = int((y - transform.f) / transform.e)
        win = ours[row - 2:row + 3, col - 2:col + 3].ravel()
        win = win[win > 0]
        rshare = g["cls"].value_counts(normalize=True)
        mshare = pd.Series(win).value_counts(normalize=True)
        sites.append({"sampleid": int(sid), "lon": round(float(g["center_x"].mean()), 5),
                      "lat": round(float(g["center_y"].mean()), 5),
                      "reference_2015_dominant": names.get(int(rshare.index[0]), "Other (bare, cloud)"),
                      "reference_2015_dominant_share": float(rshare.iloc[0]),
                      "map_2025_dominant": names[int(mshare.index[0])],
                      "map_2025_dominant_share": float(mshare.iloc[0]),
                      "forest_share_reference_2015": float(rshare.get(forest_code, 0.0)),
                      "forest_share_map_2025": float(mshare.get(forest_code, 0.0))})
    s = pd.DataFrame(sites)
    lesiv = {"n_sites": int(len(s)), "dominant_class_matches": int((s.reference_2015_dominant == s.map_2025_dominant).sum()),
             "sites": sites,
             "note": "Descriptive only: 21 sites, reference year 2015, not a sample drawn from this map."}

    out = {"compared_km2": float(valid.sum() * cell),
           "agree_with_worldcover": float((o == w).mean()), "agree_with_esri": float((o == e).mean()),
           "agree_with_both": float((agree_n == 2).mean()), "agree_with_neither": float((agree_n == 0).mean()),
           "worldcover_agree_with_esri": float((w == e).mean()),
           "areas_km2": areas, "per_class": per_class,
           "vs_worldcover_km2": area_table(o, w, codes, names, cell),
           "vs_esri_km2": area_table(o, e, codes, names, cell),
           "largest_disagreements": top, "where_worldcover_and_esri_disagree": side,
           "hansen": hansen, "lesiv2015_sites": lesiv}
    (study.OUT / "comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("vs_worldcover_km2", "vs_esri_km2", "lesiv2015_sites")}, indent=2))
    print(s[["reference_2015_dominant", "reference_2015_dominant_share", "map_2025_dominant", "map_2025_dominant_share"]])


if __name__ == "__main__":
    main()
