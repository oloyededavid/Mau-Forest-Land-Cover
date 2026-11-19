"""Training samples from map agreement.

A cell gets a training label only where ESA WorldCover 2021 and Esri land cover 2023 agree on one of the candidate
classes in study.CANDIDATE_CLASSES, and the whole 3 x 3 neighbourhood agrees on that class (to keep mixed edge
cells out). Tree cover is split with the forest management map of Lesiv et al. (2022, reference year 2015):
plantation forest where it shows planted or plantation forest (codes 31, 32), forest where it shows naturally
regenerating forest (codes 11, 20); tree cells it calls agroforestry (53) or anything else get no tree label.
Tree cells with Hansen loss in 2021-2024 are dropped, because the two maps may predate the loss. Cells with fewer
than 3 clear observations in the composite are dropped. The labels therefore come from existing maps, not from
field visits or image interpretation.

From the eligible cells a stratified random sample of up to N_PER_CLASS cells per class is drawn.

    python scripts/s03_samples.py     # writes outputs/training_samples.csv, outputs/training.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import study  # noqa: E402

N_PER_CLASS = 4000
FML_NATURAL = (11, 20)
FML_PLANTED = (31, 32)


def read_layers():
    with rasterio.open(study.DATA / "layers.tif") as src:
        names = list(src.descriptions)
        return {n: src.read(i + 1) for i, n in enumerate(names)}


def agreement_labels(wc, esri, fml, lossyear, classes=study.CANDIDATE_CLASSES):
    """Label raster (0 = no label) from map agreement, before the neighbourhood and quality filters."""
    lab = np.zeros(wc.shape, dtype=np.uint8)
    for code, c in classes.items():
        agree = np.isin(wc, c["worldcover"]) & np.isin(esri, c["esri"])
        if c["name"] == "Forest":
            agree &= np.isin(fml, FML_NATURAL)
        elif c["name"] == "Plantation forest":
            agree &= np.isin(fml, FML_PLANTED)
        if c["worldcover"] == [10]:
            agree &= ~((lossyear >= 21) & (lossyear <= 24))
        lab[agree & (lab == 0)] = code
    return lab


def homogeneous(lab, size=3):
    """Keep a label only where all cells in the size x size window carry the same label."""
    lo = ndimage.minimum_filter(lab, size=size, mode="nearest")
    hi = ndimage.maximum_filter(lab, size=size, mode="nearest")
    return np.where((lo == hi) & (lab > 0), lab, 0).astype(np.uint8)


def stratified_sample(lab, n_per_class, rng):
    """Row and column indices of a stratified random sample: up to n_per_class cells of every label > 0."""
    rows, cols, ys = [], [], []
    flat = lab.ravel()
    for code in np.unique(flat[flat > 0]):
        idx = np.flatnonzero(flat == code)
        take = idx if len(idx) <= n_per_class else rng.choice(idx, n_per_class, replace=False)
        take = np.sort(take)
        r, c = np.unravel_index(take, lab.shape)
        rows.append(r); cols.append(c); ys.append(np.full(len(take), code))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(ys)


def main():
    transform, width, height = study.grid()
    L = read_layers()
    X, valid, clear = features.read_stack()
    inbox = study.box_mask(transform, width, height)
    raw = agreement_labels(L["worldcover2021"], L["esri2023"], L["fml2015"], L["hansen_lossyear"])
    lab = homogeneous(raw)
    lab[~(valid & inbox & (clear >= 3))] = 0
    rng = np.random.default_rng(study.SEED)
    r, c, y = stratified_sample(lab, N_PER_CLASS, rng)
    xs = transform.c + (c + 0.5) * transform.a
    ys_ = transform.f + (r + 0.5) * transform.e
    lon, lat = Transformer.from_crs(study.CRS, "EPSG:4326", always_xy=True).transform(xs, ys_)
    df = pd.DataFrame({"row": r, "col": c, "x": xs, "y": ys_, "lon": np.round(lon, 6), "lat": np.round(lat, 6),
                       "label": y})
    for i, f in enumerate(study.FEATURES):
        df[f] = np.round(X[i, r, c], 5)
    df.to_csv(study.OUT / "training_samples.csv", index=False)
    info = {
        "rule": "WorldCover 2021 and Esri 2023 agree, 3 x 3 neighbourhood homogeneous, at least 3 clear observations; "
                "tree cover split by the 2015 forest management map; tree cells with Hansen loss 2021-2024 dropped",
        "n_per_class_max": N_PER_CLASS,
        "eligible_cells": {study.CANDIDATE_CLASSES[k]["name"]: int((lab == k).sum()) for k in study.CANDIDATE_CLASSES},
        "agreeing_cells_before_filters": {study.CANDIDATE_CLASSES[k]["name"]: int(((raw == k) & inbox).sum())
                                          for k in study.CANDIDATE_CLASSES},
        "sampled": {study.CANDIDATE_CLASSES[k]["name"]: int((y == k).sum()) for k in study.CANDIDATE_CLASSES},
        "box_cells": int(inbox.sum()),
        "share_of_box_with_agreement": float(((raw > 0) & inbox).sum() / inbox.sum()),
        "fml_share_of_box": {str(k): float(((L["fml2015"] == k) & inbox).sum() / inbox.sum())
                             for k in (0, 11, 20, 31, 32, 40, 53)},
        "tree_agreement_by_fml": {str(k): int(((L["worldcover2021"] == 10) & (L["esri2023"] == 2) & inbox
                                               & (L["fml2015"] == k)).sum()) for k in (0, 11, 20, 31, 32, 40, 53)},
    }
    (study.OUT / "training.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
