"""Accuracy and area estimates from the labelled reference sample (Olofsson et al., 2014).

Reads reference/labels.csv, the file exported by docs/label.html (columns id, label, ...), joins it to the map
class of each point in outputs/reference_sample.csv, and computes the error matrix (rows: map class, columns:
reference class), overall, user's and producer's accuracy and stratified area estimates with 95% intervals
(spatial_stats_core.olofsson_area_estimate). Points labelled "Unclear" are left out and counted. Points labelled
"Bare or other" are counted with cropland and grassland, because the map has no bare class (bare land is under
0.1% of the box in WorldCover 2021); their number is reported.

Until every point has a label, the result is written as pending and nothing is estimated.

    python scripts/s09_accuracy.py [path/to/labels.csv]    # writes outputs/accuracy.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from spatial_stats_core import olofsson_area_estimate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

LABELS = study.ROOT / "reference" / "labels.csv"
UNCLEAR = "Unclear"
OTHER = "Bare or other"
OTHER_TO = "Cropland and grassland"


def error_matrix(map_cls, ref_cls, names):
    idx = {n: i for i, n in enumerate(names)}
    m = np.zeros((len(names), len(names)), dtype=int)
    for a, b in zip(map_cls, ref_cls):
        m[idx[a], idx[b]] += 1
    return m


def estimate(sample, labels, names, mapped_km2):
    """Return the accuracy block, or a pending block when labels are missing."""
    d = sample.merge(labels[["id", "label"]], on="id", how="left")
    n_total = len(d)
    done = d["label"].notna() & (d["label"].astype(str).str.len() > 0)
    if not done.all():
        return {"status": "pending", "n_sample": n_total, "n_labelled": int(done.sum())}
    bad = set(d["label"]) - set(names) - {UNCLEAR, OTHER}
    if bad:
        raise SystemExit(f"labels outside the class list: {sorted(bad)}")
    n_unclear = int((d["label"] == UNCLEAR).sum())
    n_other = int((d["label"] == OTHER).sum())
    d = d[d["label"] != UNCLEAR].copy()
    d.loc[d["label"] == OTHER, "label"] = OTHER_TO
    m = error_matrix(d["map_class"], d["label"], names)
    areas = np.array([mapped_km2[n] for n in names])
    r = olofsson_area_estimate(m, areas, class_names=names)
    t = r.class_table.set_index("class_name")
    return {
        "status": "complete", "n_sample": n_total, "n_used": int(len(d)), "n_unclear": n_unclear,
        "n_bare_or_other": n_other, "error_matrix": m.tolist(), "classes": names,
        "overall_accuracy": r.overall_accuracy, "overall_accuracy_ci95": list(r.overall_accuracy_ci),
        "per_class": {n: {k: float(t.loc[n, k]) for k in
                          ("mapped_area", "estimated_area", "area_ci_halfwidth", "users_accuracy",
                           "users_ci_halfwidth", "producers_accuracy", "producers_ci_halfwidth")} for n in names},
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LABELS
    sample = pd.read_csv(study.OUT / "reference_sample.csv")
    classes = json.loads((study.OUT / "classes.json").read_text())
    names = [c["name"] for c in classes["final"]]
    mapped = {n: v["area_km2"] for n, v in json.loads((study.OUT / "map_areas.json").read_text())["mapped"].items()}
    if path.exists():
        labels = pd.read_csv(path, dtype=str)
        res = estimate(sample, labels, names, mapped)
        res["labels_file"] = str(path.relative_to(study.ROOT)) if path.is_relative_to(study.ROOT) else str(path)
    else:
        res = {"status": "pending", "n_sample": int(len(sample)), "n_labelled": 0}
    res["note"] = ("Reference labels are visual interpretation of high-resolution imagery by one interpreter, "
                   "not field data.")
    (study.OUT / "accuracy.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
