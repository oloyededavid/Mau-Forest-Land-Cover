"""Train the random forest on all training samples with the final class scheme and classify the whole grid.

    python scripts/s05_classify.py   # writes data/landcover_2025.tif (class, top-class probability x 100),
                                     # outputs/map_areas.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import s04_cv  # noqa: E402
import study  # noqa: E402


def final_labels(df, classes):
    """Relabel the training sample to the final scheme (merged classes take the code they were merged into)."""
    keep = {c["code"] for c in classes["final"]}
    y = df["label"].to_numpy().copy()
    if not classes["plantation_decision"]["kept"]:
        plant = [k for k, v in study.CANDIDATE_CLASSES.items() if v["name"] == "Plantation forest"][0]
        forest = [k for k, v in study.CANDIDATE_CLASSES.items() if v["name"] == "Forest"][0]
        y[y == plant] = forest
    assert set(np.unique(y)) <= keep, "training labels outside the final scheme"
    return y


def main():
    classes = json.loads((study.OUT / "classes.json").read_text())
    df = pd.read_csv(study.OUT / "training_samples.csv")
    y = final_labels(df, classes)
    model = s04_cv.models()["random_forest"]
    model.fit(df[study.FEATURES].to_numpy(np.float32), y)
    imp = dict(zip(study.FEATURES, model.feature_importances_.round(4).tolist()))

    transform, width, height = study.grid()
    inbox = study.box_mask(transform, width, height)
    profile = dict(driver="GTiff", dtype="uint8", width=width, height=height, count=2, crs=study.CRS,
                   transform=transform, nodata=0, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    counts = {c["code"]: 0 for c in classes["final"]}
    prob_sum = {c["code"]: 0.0 for c in classes["final"]}
    t0 = time.time()
    with rasterio.open(study.DATA / "landcover_2025.tif", "w", **profile) as dst:
        for r0 in range(0, height, 256):
            win = Window(0, r0, width, min(256, height - r0))
            X, valid, _ = features.read_stack(win)
            ok = valid & inbox[r0:r0 + win.height]
            cls = np.zeros(ok.shape, dtype=np.uint8)
            prob = np.zeros(ok.shape, dtype=np.uint8)
            if ok.any():
                P = model.predict_proba(X[:, ok].T)
                k = P.argmax(axis=1)
                cls[ok] = model.classes_[k]
                prob[ok] = np.round(P.max(axis=1) * 100).astype(np.uint8)
                for code in counts:
                    sel = cls[ok] == code
                    counts[code] += int(sel.sum())
                    prob_sum[code] += float(prob[ok][sel].sum())
            dst.write(cls, 1, window=win)
            dst.write(prob, 2, window=win)
            print(f"rows {r0}-{r0 + win.height} {time.time() - t0:.0f} s", flush=True)
        dst.set_band_description(1, "class")
        dst.set_band_description(2, "top_class_probability_pct")

    cell_km2 = study.RES * study.RES / 1e6
    total = sum(counts.values())
    names = {c["code"]: c["name"] for c in classes["final"]}
    out = {
        "model": "random forest, 300 trees, min_samples_leaf 2, trained on all training samples",
        "cell_km2": cell_km2,
        "classified_cells": total,
        "box_cells": int(inbox.sum()),
        "unclassified_cells_in_box": int(inbox.sum() - total),
        "mapped": {names[k]: {"code": k, "cells": counts[k], "area_km2": counts[k] * cell_km2,
                              "share": counts[k] / total,
                              "mean_top_probability": prob_sum[k] / max(counts[k], 1) / 100} for k in counts},
        "feature_importance": imp,
    }
    (study.OUT / "map_areas.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
