"""Cross-validation of the classifiers on the map-agreement training sample, random against spatially blocked.

Models: random forest (Breiman, 2001) and histogram gradient boosting (scikit-learn), fixed settings, no tuning.
Schemes: random stratified 5-fold, and spatial block 5-fold (spatial_stats_core.SpatialBlockKFold, stratified by
class over blocks) with square blocks of 5, 10 and 20 km on the UTM grid. Out-of-fold predictions are pooled over
the five folds before computing precision, recall and accuracy.

Class decision (fixed before running): plantation forest stays a separate class only if, under 10 km blocked CV
with the random forest, both its precision and its recall reach 0.70. Otherwise its samples are relabelled as
forest. A candidate class with no training cells is dropped. The final scheme is then cross-validated again.

Important: these accuracies measure agreement with labels taken from the two existing maps, on cells where both
maps agree and the neighbourhood is uniform. Those are the easier cells, so the numbers are not map accuracy.

    python scripts/s04_cv.py      # writes outputs/cv.json, outputs/classes.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from spatial_stats_core import SpatialBlockKFold, summarize_fold_distances

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

BLOCKS_KM = (5, 10, 20)
DECISION_BLOCK_KM = 10
THRESHOLD = 0.70
N_SPLITS = 5


def models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                                n_jobs=4, random_state=study.SEED),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                            random_state=study.SEED),
    }


def splits(scheme, X, y, xy):
    if scheme == "random":
        return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=study.SEED).split(X, y))
    km = int(scheme.split("_")[1].removesuffix("km"))
    cv = SpatialBlockKFold(n_splits=N_SPLITS, block_size=km * 1000.0, stratify=True, random_state=study.SEED)
    return list(cv.split(X, y, coords=xy))


def scores(y, pred, labels, names):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return {
        "overall_accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {names[k]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                 "n": int(n[i])} for i, k in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def run(df, labels, names):
    X = df[study.FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    out = {}
    for scheme in ["random"] + [f"block_{k}km" for k in BLOCKS_KM]:
        sp = splits(scheme, X, y, xy)
        dist = summarize_fold_distances(xy, sp)
        out[scheme] = {"fold_distances_m": dist, "test_sizes": [int(len(t)) for _, t in sp], "models": {}}
        for mname, model in models().items():
            pred = np.zeros_like(y)
            fold_oa = []
            for tr, te in sp:
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])
                fold_oa.append(float(accuracy_score(y[te], pred[te])))
            s = scores(y, pred, labels, names)
            s["fold_overall_accuracy"] = fold_oa
            out[scheme]["models"][mname] = s
            print(f"{scheme:12s} {mname:18s} OA {s['overall_accuracy']:.3f} macro-F1 {s['macro_f1']:.3f}", flush=True)
    return out


def main():
    df = pd.read_csv(study.OUT / "training_samples.csv")
    cand = {k: v["name"] for k, v in study.CANDIDATE_CLASSES.items()}
    present = sorted(int(k) for k in df["label"].unique())
    dropped = {cand[k]: "no training cells where both maps agree in a uniform 3 x 3 neighbourhood"
               for k in cand if k not in present}

    stage1 = run(df, present, cand)
    plant = [k for k, v in cand.items() if v == "Plantation forest"][0]
    forest = [k for k, v in cand.items() if v == "Forest"][0]
    dec = stage1[f"block_{DECISION_BLOCK_KM}km"]["models"]["random_forest"]["per_class"].get(cand[plant])
    keep_plant = dec is not None and dec["precision"] >= THRESHOLD and dec["recall"] >= THRESHOLD
    final = dict(cand)
    df2 = df.copy()
    if not keep_plant:
        df2.loc[df2["label"] == plant, "label"] = forest
        final.pop(plant)
        final[forest] = "Forest (natural and plantation)"
    for k in list(final):
        if k not in present:
            final.pop(k)
    labels = sorted(final)
    stage2 = run(df2, labels, final) if not keep_plant else stage1

    classes = {
        "final": [{"code": k, "name": final[k], "colour": study.CANDIDATE_CLASSES[k]["colour"]} for k in labels],
        "dropped": dropped,
        "plantation_decision": {
            "rule": f"keep if precision and recall >= {THRESHOLD} under {DECISION_BLOCK_KM} km blocked CV (random forest)",
            "threshold": THRESHOLD, "block_km": DECISION_BLOCK_KM,
            "precision": None if dec is None else dec["precision"],
            "recall": None if dec is None else dec["recall"],
            "kept": bool(keep_plant),
        },
    }
    (study.OUT / "classes.json").write_text(json.dumps(classes, indent=2))
    (study.OUT / "cv.json").write_text(json.dumps({
        "n_samples": int(len(df)), "n_splits": N_SPLITS, "blocks_km": list(BLOCKS_KM),
        "note": "Accuracy against map-agreement labels on uniform cells; not map accuracy.",
        "candidate_scheme": {"classes": [cand[k] for k in present], "results": stage1},
        "final_scheme": {"classes": [final[k] for k in labels], "results": stage2},
    }, indent=2))
    print(json.dumps(classes, indent=2))


if __name__ == "__main__":
    main()
