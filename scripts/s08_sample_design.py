"""Stratified random reference sample for the 2025 map, to be labelled by eye on high-resolution imagery with
docs/label.html.

Strata are the map classes. The total sample size follows Olofsson et al. (2014, eq. 13; Cochran, 1977):
n = (sum_i W_i S_i / S(O))^2, with W_i the mapped area share of class i, S_i = sqrt(U_i (1 - U_i)) for an
expected user's accuracy U_i, and S(O) the target standard error of overall accuracy. Each class gets its
proportional share, raised to a minimum for small classes (spatial_stats_core.allocate_stratified_sample).
The expected user's accuracies below are assumptions, not results.

The labelling page gets only an id and a location for each point, in random order, never the map class, so the
interpreter labels blind.

    python scripts/s08_sample_design.py   # writes outputs/reference_sample.csv, outputs/sample_design.json,
                                          # docs/data/label_points.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from spatial_stats_core import allocate_stratified_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

TARGET_SE = 0.015
MIN_PER_CLASS = 75
EXPECTED_UA = {"Forest (natural and plantation)": 0.90, "Forest": 0.90, "Plantation forest": 0.70,
               "Cropland and grassland": 0.85, "Built-up": 0.70, "Water and wetland": 0.90}


def draw(cls, codes, n_per, rng, inbox):
    """Random cell indices per stratum (without replacement)."""
    out = []
    flat = cls.ravel()
    ok = inbox.ravel()
    for code, n in zip(codes, n_per):
        idx = np.flatnonzero((flat == code) & ok)
        take = rng.choice(idx, int(n), replace=False)
        out.append(pd.DataFrame({"cell": take, "map_code": code}))
    return pd.concat(out, ignore_index=True)


def main():
    classes = json.loads((study.OUT / "classes.json").read_text())
    areas = json.loads((study.OUT / "map_areas.json").read_text())["mapped"]
    names = [c["name"] for c in classes["final"]]
    codes = [c["code"] for c in classes["final"]]
    cells = np.array([areas[n]["cells"] for n in names], dtype=float)
    ua = np.array([EXPECTED_UA[n] for n in names])
    alloc = allocate_stratified_sample(cells, ua, target_overall_se=TARGET_SE, min_per_class=MIN_PER_CLASS)
    alloc_01 = allocate_stratified_sample(cells, ua, target_overall_se=0.01, min_per_class=MIN_PER_CLASS)

    transform, width, height = study.grid()
    inbox = study.box_mask(transform, width, height)
    with rasterio.open(study.DATA / "landcover_2025.tif") as s:
        cls = s.read(1)
    rng = np.random.default_rng(study.SEED + 1)
    df = draw(cls, codes, alloc["recommended_allocation"], rng, inbox)
    r, c = np.unravel_index(df["cell"].to_numpy(), cls.shape)
    df["x"] = transform.c + (c + 0.5) * transform.a
    df["y"] = transform.f + (r + 0.5) * transform.e
    lon, lat = Transformer.from_crs(study.CRS, "EPSG:4326", always_xy=True).transform(df["x"], df["y"])
    df["lon"] = np.round(lon, 6); df["lat"] = np.round(lat, 6)
    df["map_class"] = df["map_code"].map(dict(zip(codes, names)))
    df = df.sample(frac=1.0, random_state=study.SEED).reset_index(drop=True)   # random order, strata mixed
    df.insert(0, "id", [f"P{i + 1:04d}" for i in range(len(df))])
    df[["id", "lon", "lat", "x", "y", "map_code", "map_class"]].to_csv(study.OUT / "reference_sample.csv", index=False)

    (study.DOCS / "data").mkdir(parents=True, exist_ok=True)
    label_classes = names + ["Bare or other", "Unclear"]
    (study.DOCS / "data" / "label_points.json").write_text(json.dumps({
        "note": "Blind reference sample: map classes are not included. Label each point from the imagery.",
        "classes": label_classes,
        "points": [{"id": i, "lon": float(a), "lat": float(b)} for i, a, b in zip(df["id"], df["lon"], df["lat"])],
    }, separators=(",", ":")))
    design = {
        "target_overall_se": TARGET_SE, "min_per_class": MIN_PER_CLASS,
        "expected_users_accuracy": dict(zip(names, ua.tolist())),
        "mapped_share": dict(zip(names, (cells / cells.sum()).tolist())),
        "formula_sample_size": alloc["formula_sample_size"],
        "total_sample_size": alloc["total_sample_size"],
        "allocation": dict(zip(names, [int(v) for v in alloc["recommended_allocation"]])),
        "proportional_allocation": dict(zip(names, [int(v) for v in alloc["proportional_allocation"]])),
        "formula_sample_size_at_se_0_01": alloc_01["formula_sample_size"],
        # expected 95% half-width of each user's accuracy with the allocated points, if U_i holds
        "expected_users_ci_halfwidth": dict(zip(names, (1.959964 * np.sqrt(ua * (1 - ua) / (np.asarray(
            alloc["recommended_allocation"], dtype=float) - 1))).tolist())),
        "label_classes": label_classes,
    }
    (study.OUT / "sample_design.json").write_text(json.dumps(design, indent=2))
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
