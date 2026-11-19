"""Accuracy and area estimates from the labelled reference sample (Olofsson et al., 2014).

Reads reference/labels.csv, the file exported by docs/label.html (columns id, label, ...), joins it to the map
class of each point in outputs/reference_sample.csv, and computes the error matrix (rows: map class, columns:
reference class), overall, user's and producer's accuracy and stratified area estimates with 95% intervals
(spatial_stats_core.olofsson_area_estimate). Points labelled "Unclear" are left out and counted. The map has no
bare class, so a point labelled "Bare or other" is an extra reference-only column: it counts as an error for
whatever map class it falls in, and the area of bare or other land is estimated like any other reference class.
When that column is used, the same Olofsson et al. (2014) equations are applied to the rectangular matrix
(olofsson_rect); with an empty column they give the same result as the package.

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


def error_matrix(map_cls, ref_cls, names):
    idx = {n: i for i, n in enumerate(names)}
    m = np.zeros((len(names), len(names)), dtype=int)
    for a, b in zip(map_cls, ref_cls):
        m[idx[a], idx[b]] += 1
    return m


def olofsson_rect(n, areas, z=1.959964):
    """Olofsson et al. (2014) estimators for q map strata (rows) and q + r reference classes (columns); the first
    q columns are the map classes in the same order. Returns overall accuracy and its interval, and for every
    reference class the estimated area with interval and, for the map classes, user's and producer's accuracy."""
    n = np.asarray(n, dtype=float)
    q = n.shape[0]
    W = np.asarray(areas, dtype=float) / np.sum(areas)
    ni = n.sum(axis=1)
    if np.any(ni < 2):
        raise ValueError("each map stratum needs at least 2 points")
    p = W[:, None] * n / ni[:, None]
    v = p * (W[:, None] - p) / (ni - 1)[:, None]
    pk, vk = p.sum(axis=0), v.sum(axis=0)
    oa = float(np.trace(p[:, :q]))
    oa_se = float(np.sqrt(sum(v[i, i] for i in range(q))))
    ua = np.diag(n[:, :q]) / ni
    ua_se = np.sqrt(ua * (1 - ua) / (ni - 1))
    pa = np.array([p[j, j] / pk[j] if pk[j] > 0 else 0.0 for j in range(q)])
    pa_var = np.array([((1 - pa[j]) ** 2 * v[j, j] + pa[j] ** 2 * (vk[j] - v[j, j])) / pk[j] ** 2 if pk[j] > 0
                       else 0.0 for j in range(q)])
    total = float(np.sum(areas))
    return {"overall_accuracy": oa, "overall_accuracy_ci95": [oa - z * oa_se, oa + z * oa_se],
            "estimated_area": pk * total, "area_ci_halfwidth": z * np.sqrt(vk) * total,
            "users_accuracy": ua, "users_ci_halfwidth": z * ua_se,
            "producers_accuracy": pa, "producers_ci_halfwidth": z * np.sqrt(pa_var)}


def estimate(sample, labels, names, mapped_km2):
    """Return the accuracy block, or a pending block when labels are missing."""
    if labels["id"].duplicated().any():
        raise SystemExit(f"labels file has duplicate ids: {sorted(labels.loc[labels['id'].duplicated(), 'id'])[:5]}")
    if not sample["id"].is_unique:
        raise SystemExit("reference sample has duplicate ids")
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
    areas = np.array([mapped_km2[n] for n in names])
    out = {"status": "complete", "n_sample": n_total, "n_used": int(len(d)), "n_unclear": n_unclear,
           "n_bare_or_other": n_other, "classes": names}
    keys = ("estimated_area", "area_ci_halfwidth", "users_accuracy", "users_ci_halfwidth", "producers_accuracy",
            "producers_ci_halfwidth")
    if n_other == 0:
        m = error_matrix(d["map_class"], d["label"], names)
        r = olofsson_area_estimate(m, areas, class_names=names)
        t = r.class_table.set_index("class_name")
        out.update({"error_matrix": m.tolist(), "reference_classes": names,
                    "overall_accuracy": r.overall_accuracy, "overall_accuracy_ci95": list(r.overall_accuracy_ci),
                    "per_class": {n: {"mapped_area": float(t.loc[n, "mapped_area"]),
                                      **{k: float(t.loc[n, k]) for k in keys}} for n in names}})
        return out
    ref = names + [OTHER]
    m = np.zeros((len(names), len(ref)), dtype=int)
    for a, b in zip(d["map_class"], d["label"]):
        m[names.index(a), ref.index(b)] += 1
    r = olofsson_rect(m, areas)
    out.update({"error_matrix": m.tolist(), "reference_classes": ref,
                "overall_accuracy": r["overall_accuracy"], "overall_accuracy_ci95": r["overall_accuracy_ci95"],
                "per_class": {n: {"mapped_area": float(areas[i]), **{k: float(r[k][i]) for k in keys}}
                              for i, n in enumerate(names)},
                "bare_or_other": {"estimated_area": float(r["estimated_area"][-1]),
                                  "area_ci_halfwidth": float(r["area_ci_halfwidth"][-1])}})
    return out


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
