"""Build paper/paper.md from paper/template.md. Every number comes from the JSON and CSV files in outputs/, which
the scripts s01 to s10 write. The build stops if a statement in the template no longer matches the outputs, or
if a placeholder has no value.

    python scripts/build_paper.py && pandoc paper/paper.md -o paper/paper.docx --resource-path=. && python scripts/build_page.py
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

ROOT = study.ROOT
OUT = study.OUT


def load(name):
    return json.loads((OUT / name).read_text())


TITLE_BASE = ("Mapping land cover in the Mau Forest Complex, Kenya, with Sentinel-2: a classifier tested by spatially "
              "blocked cross-validation")
TITLE = {"pending": TITLE_BASE + ", and a stratified sample design for accuracy and area estimation",
         "complete": TITLE_BASE + ", with stratified area estimates for 2025"}


def title(acc):
    """The title claims area estimates only once the labelled reference sample has produced them."""
    return TITLE["complete" if acc["status"] == "complete" else "pending"]


def reweighted_precision(cm, shares):
    """Precision when the classes occur in proportion `shares` instead of the balanced test folds.
    cm: rows true class, columns predicted class (counts)."""
    import numpy as np
    cm = np.asarray(cm, dtype=float)
    r = cm / cm.sum(axis=1, keepdims=True)          # P(predicted j | true i)
    w = np.asarray(shares, dtype=float)
    col = (w[:, None] * r).sum(axis=0)
    return np.where(col > 0, w * np.diag(r) / col, 0.0)


def need(cond, msg):
    if not cond:
        raise SystemExit("template statement no longer true: " + msg)


def p2(x):
    """A proportion as 0.00."""
    return f"{x:.2f}"


def pct(x, d=0):
    return f"{x * 100:.{d}f}"


def km(x):
    return f"{x:,.0f}"


def long_date(iso):
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def cv_table(cv):
    rows = ["| Scheme | Folds | Overall agreement | Macro F1 | Forest precision | Forest recall | Plantation recall |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for scheme_key, label in (("candidate_scheme", "With plantation class"), ("final_scheme", "Final (plantation merged)")):
        res = cv[scheme_key]["results"]
        for k, fold in (("random", "Random"), ("block_5km", "5 km blocks"), ("block_10km", "10 km blocks"),
                        ("block_20km", "20 km blocks")):
            m = res[k]["models"]["random_forest"]
            pc = m["per_class"]
            forest = next(v for n, v in pc.items() if n.startswith("Forest"))
            plant = pc.get("Plantation forest")
            rows.append(f"| {label} | {fold} | {p2(m['overall_accuracy'])} | {p2(m['macro_f1'])} | "
                        f"{p2(forest['precision'])} | {p2(forest['recall'])} | "
                        f"{p2(plant['recall']) if plant else '–'} |")
    return "\n".join(rows)


def area_table(areas, comp):
    rows = ["| Class | This map (km²) | Share of box | WorldCover 2021 (km²) | Esri 2023 (km²) |",
            "| :--- | ---: | ---: | ---: | ---: |"]
    for name, v in areas["mapped"].items():
        c = comp["areas_km2"][name]
        rows.append(f"| {name} | {km(v['area_km2'])} | {pct(v['share'], 1)}% | {km(c['worldcover2021'])} | "
                    f"{km(c['esri2023'])} |")
    o = comp["areas_km2"]["Other (bare, cloud)"]
    rows.append(f"| Bare or other | – | – | {o['worldcover2021']:.1f} | {o['esri2023']:.1f} |")
    return "\n".join(rows)


def accuracy_blocks(acc, design):
    if acc["status"] != "complete":
        n = acc.get("n_labelled", 0)
        sec = (f"Pending the reference sample. The {design['total_sample_size']} points of the stratified sample "
               f"(Section 2.6) have not yet been labelled ({n} of {acc['n_sample']} so far), so this paper reports "
               "no accuracy or area estimates with confidence intervals, and the class areas in Table 2 are mapped "
               "areas, which can be biased by map errors. When the labels are added, "
               "`scripts/s09_accuracy.py` computes the error matrix, overall, user's and producer's accuracy and "
               "stratified area estimates with 95% intervals, and this section is rebuilt from its output.")
        return {"accuracy_section": sec,
                "abstract_accuracy": "Accuracy and stratified area estimates are pending the labelling of a "
                                     f"{design['total_sample_size']}-point stratified reference sample, for which "
                                     "the study provides the sample and a labelling tool; no open reference data "
                                     "set had enough points inside the box.",
                "conclusion_accuracy": "Accuracy and area estimates with confidence intervals await the labelled "
                                       "reference sample."}
    rows = ["| Class | Mapped (km²) | Estimated (km², 95% interval) | User's accuracy | Producer's accuracy |",
            "| :--- | ---: | ---: | ---: | ---: |"]
    for n, v in acc["per_class"].items():
        rows.append(f"| {n} | {km(v['mapped_area'])} | {km(v['estimated_area'])} ± {km(v['area_ci_halfwidth'])} | "
                    f"{p2(v['users_accuracy'])} ± {p2(v['users_ci_halfwidth'])} | "
                    f"{p2(v['producers_accuracy'])} ± {p2(v['producers_ci_halfwidth'])} |")
    oa, ci = acc["overall_accuracy"], acc["overall_accuracy_ci95"]
    sec = (f"From {acc['n_used']} labelled points ({acc['n_unclear']} left out as unclear; {acc['n_bare_or_other']} "
           f"labelled bare or other, counted as errors of their map class), overall accuracy is {p2(oa)} (95% interval "
           f"{p2(ci[0])} to {p2(ci[1])}). Table 3 gives the stratified area estimates (Olofsson et al., 2014). The "
           "labels are one interpreter's reading of high-resolution imagery.\n\nTable 3. Accuracy and stratified "
           "area estimates from the reference sample, with 95% intervals.\n\n" + "\n".join(rows))
    return {"accuracy_section": sec,
            "abstract_accuracy": f"From a {acc['n_sample']}-point stratified reference sample labelled on "
                                 f"high-resolution imagery, overall accuracy is {p2(oa)} ({p2(ci[0])} to {p2(ci[1])}).",
            "conclusion_accuracy": f"Overall accuracy from the reference sample is {p2(oa)}."}


def build():
    comp_meta = load("composite.json")
    tr = load("training.json")
    cv = load("cv.json")
    classes = load("classes.json")
    areas = load("map_areas.json")
    comp = load("comparison.json")
    H = load("hansen_loss_by_year.json")
    ref = load("reference_search.json")
    design = load("sample_design.json")
    acc = load("accuracy.json")

    cand = cv["candidate_scheme"]["results"]
    fin = cv["final_scheme"]["results"]
    rf = lambda r, k: r[k]["models"]["random_forest"]  # noqa: E731
    gb = lambda r, k: r[k]["models"]["gradient_boosting"]  # noqa: E731
    dec = classes["plantation_decision"]
    names = [c["name"] for c in classes["final"]]
    forest_name = next(n for n in names if n.startswith("Forest"))
    M = areas["mapped"]

    # statements the template makes
    need(not dec["kept"] and forest_name == "Forest (natural and plantation)", "plantation class merged with forest")
    need(dec["recall"] < dec["precision"], "plantation fails on recall")
    need("Bare" in classes["dropped"], "no bare class trained")
    need(names == ["Forest (natural and plantation)", "Cropland and grassland", "Built-up", "Water and wetland"],
         "final classes are forest, cropland and grassland, built-up, water and wetland")
    s = tr["sampled"]
    need(s["Forest"] + s["Plantation forest"] == 2 * s["Cropland and grassland"] == 2 * s["Built-up"]
         == 2 * s["Water and wetland"], "merged forest class has twice the training cells of each other class")
    need(rf(cand, "random")["overall_accuracy"] - rf(cand, "block_10km")["overall_accuracy"] > 0.03,
         "blocked CV lowers accuracy with the plantation class kept")
    cpr = rf(cand, "random")["per_class"]; cpb = rf(cand, "block_10km")["per_class"]
    need(cpb["Plantation forest"]["recall"] < cpr["Plantation forest"]["recall"] - 0.1, "plantation recall falls")
    need(cpb["Forest"]["precision"] < cpr["Forest"]["precision"] - 0.1, "forest precision falls")
    cm = rf(cand, "block_10km")["confusion_matrix"]
    cls_c = cv["candidate_scheme"]["classes"]
    i_p, i_f = cls_c.index("Plantation forest"), cls_c.index("Forest")
    need(cm[i_p][i_f] == max(v for j, v in enumerate(cm[i_p]) if j != i_p), "plantation cells are predicted as forest")
    finals = [rf(fin, k)["overall_accuracy"] for k in fin]
    need(max(finals) - min(finals) < 0.03, "final scheme: little change between random and blocked")
    diffs = [abs(rf(fin, k)["overall_accuracy"] - gb(fin, k)["overall_accuracy"]) for k in fin]
    need(max(diffs) < 0.02, "the two models are close")
    built_prob = M["Built-up"]["mean_top_probability"]
    need(built_prob == min(v["mean_top_probability"] for v in M.values()), "built-up is the least certain class")
    need(comp["per_class"]["Built-up"]["share_agreeing_with_both"] == min(v["share_agreeing_with_both"]
                                                                          for v in comp["per_class"].values()),
         "built-up is where the maps differ most")
    A = comp["areas_km2"]
    need(A[forest_name]["worldcover2021"] > A[forest_name]["this_map"] > A[forest_name]["esri2023"],
         "forest area lies between WorldCover and Esri")
    need(A["Built-up"]["esri2023"] > A["Built-up"]["worldcover2021"] and
         A["Cropland and grassland"]["esri2023"] > A["Cropland and grassland"]["worldcover2021"],
         "Esri has more crops and built-up land than WorldCover")
    pcmp = comp["per_class"]
    need(pcmp[forest_name]["share_agreeing_with_worldcover"] > pcmp[forest_name]["share_agreeing_with_esri"],
         "forest closer to WorldCover")
    need(pcmp["Cropland and grassland"]["share_agreeing_with_esri"] >
         pcmp["Cropland and grassland"]["share_agreeing_with_worldcover"], "cropland closer to Esri")
    side = comp["where_worldcover_and_esri_disagree"]
    need(side["with_esri"] > side["with_worldcover"], "sides with Esri where the maps disagree")
    t1, t2 = comp["largest_disagreements"][:2]
    need((t1["this_map"], t1["worldcover2021"], t1["esri2023"]) ==
         ("Cropland and grassland", forest_name, "Cropland and grassland"), "largest disagreement")
    need((t2["this_map"], t2["worldcover2021"], t2["esri2023"]) == (forest_name, forest_name, "Cropland and grassland"),
         "second disagreement")
    need(not ref["enough_for_accuracy"] and ref["lesiv2025"]["sites_in_box"] < ref["target_points"]
         and ref["fritz2017"]["locations_in_box"] < ref["target_points"], "reference sets too small")
    les = comp["lesiv2015_sites"]
    doms = [s["reference_2015_dominant"] for s in les["sites"]]
    need(doms.count("Cropland and grassland") > len(doms) / 2, "Lesiv sites are mostly cropland")
    need(comp["hansen"]["share_of_loss_2001_2024_mapped_forest_2025"] < 0.5, "most loss cells are not forest now")
    need(comp_meta["clear_count_in_box"]["share_zero"] < 0.001, "almost no cells without a clear observation")
    need(design["formula_sample_size"] < design["total_sample_size"], "minimum per class raises the sample")
    small = [n for n in names if design["allocation"][n] == design["min_per_class"]]
    need(set(small) == {"Built-up", "Water and wetland"}, "the small classes are raised to the minimum")

    shares = [M[n]["share"] for n in names]
    rw = reweighted_precision(rf(fin, "block_10km")["confusion_matrix"], shares)
    built_rw = float(rw[names.index("Built-up")])
    need(built_rw < fpc_built_precision(fin, rf), "reweighting lowers built-up precision")
    n_train = sum(s.values())
    years = H["loss_km2_by_year"]
    ranked = sorted(years.items(), key=lambda kv: kv[1], reverse=True)
    fold20 = rf(fin, "block_20km")["fold_overall_accuracy"]
    fpc = rf(fin, "block_10km")["per_class"]
    box_km2 = areas["box_cells"] * areas["cell_km2"]
    vals = {
        "date": f"{date.today().strftime('%B')} {date.today().year}",
        "box_km2": km(box_km2),
        "lon0": f"{study.BBOX[0]:.2f}", "lon1": f"{study.BBOX[2]:.2f}",
        "lat0": f"{abs(study.BBOX[3]):.2f}", "lat1": f"{abs(study.BBOX[1]):.2f}",
        "n_scenes": str(comp_meta["n_scenes"]), "n_acq": str(comp_meta["n_acquisitions"]),
        "first_date_long": long_date(comp_meta["first_date"]), "last_date_long": long_date(comp_meta["last_date"]),
        "max_cloud": str(comp_meta["max_cloud_pct"]),
        "clear_median": f"{comp_meta['clear_count_in_box']['median']:.0f}",
        "clear_p05": f"{comp_meta['clear_count_in_box']['p05']:.0f}",
        "clear_zero_pct": f"{comp_meta['clear_count_in_box']['share_zero'] * 100:.3f}",
        "n_features": str(len(study.FEATURES)),
        "share_agree_pct": pct(tr["share_of_box_with_agreement"]),
        "fml_agro_pct": pct(tr["fml_share_of_box"]["53"]),
        "n_per_class": f"{tr['n_per_class_max']:,}", "n_train": f"{sum(tr['sampled'].values()):,}",
        "elig_forest": f"{tr['eligible_cells']['Forest']:,}", "elig_crop": f"{tr['eligible_cells']['Cropland and grassland']:,}",
        "elig_built": f"{tr['eligible_cells']['Built-up']:,}", "elig_water": f"{tr['eligible_cells']['Water and wetland']:,}",
        "elig_plant": f"{tr['eligible_cells']['Plantation forest']:,}",
        "agree_bare": str(tr["agreeing_cells_before_filters"]["Bare"]),
        "threshold": p2(dec["threshold"]),
        "plant_rec": p2(dec["recall"]),
        "c_rf_random": p2(rf(cand, "random")["overall_accuracy"]), "c_rf_block10": p2(rf(cand, "block_10km")["overall_accuracy"]),
        "c_plant_rec_random": p2(cpr["Plantation forest"]["recall"]),
        "c_forest_prec_random": p2(cpr["Forest"]["precision"]), "c_forest_prec_block10": p2(cpb["Forest"]["precision"]),
        "f_rf_random": p2(rf(fin, "random")["overall_accuracy"]), "f_rf_block5": p2(rf(fin, "block_5km")["overall_accuracy"]),
        "f_rf_block10": p2(rf(fin, "block_10km")["overall_accuracy"]), "f_rf_block20": p2(rf(fin, "block_20km")["overall_accuracy"]),
        "f_gb_random": p2(gb(fin, "random")["overall_accuracy"]), "f_gb_block10": p2(gb(fin, "block_10km")["overall_accuracy"]),
        "f_model_diff": f"{max(diffs):.3f}",
        "f_forest_p": p2(fpc[forest_name]["precision"]), "f_forest_r": p2(fpc[forest_name]["recall"]),
        "f_crop_p": p2(fpc["Cropland and grassland"]["precision"]), "f_crop_r": p2(fpc["Cropland and grassland"]["recall"]),
        "f_built_p": p2(fpc["Built-up"]["precision"]), "f_built_r": p2(fpc["Built-up"]["recall"]),
        "f_water_p": p2(fpc["Water and wetland"]["precision"]), "f_water_r": p2(fpc["Water and wetland"]["recall"]),
        "f_fold20_min": p2(min(fold20)), "f_fold20_max": p2(max(fold20)),
        "dist_min_10": f"{fin['block_10km']['fold_distances_m']['min_train_test_distance']:.0f}",
        "dist_med_10": f"{fin['block_10km']['fold_distances_m']['mean_fold_median_distance']:,.0f}",
        "dist_med_random": f"{fin['random']['fold_distances_m']['mean_fold_median_distance']:.0f}",
        "cv_table": cv_table(cv), "area_table": area_table(areas, comp),
        "forest_km2": km(M[forest_name]["area_km2"]), "forest_pct": pct(M[forest_name]["share"], 1),
        "built_km2": km(M["Built-up"]["area_km2"]),
        "prob_forest": p2(M[forest_name]["mean_top_probability"]), "prob_built": p2(built_prob),
        "wc_forest_km2": km(A[forest_name]["worldcover2021"]), "es_forest_km2": km(A[forest_name]["esri2023"]),
        "wc_built_km2": km(A["Built-up"]["worldcover2021"]), "es_built_km2": km(A["Built-up"]["esri2023"]),
        "agree_wc_pct": pct(comp["agree_with_worldcover"]), "agree_es_pct": pct(comp["agree_with_esri"]),
        "agree_both_pct": pct(comp["agree_with_both"]), "agree_neither_pct": pct(comp["agree_with_neither"]),
        "wc_es_pct": pct(comp["worldcover_agree_with_esri"]), "dis_km2": km(side["cells_km2"]),
        "side_es_pct": pct(side["with_esri"]), "side_wc_pct": pct(side["with_worldcover"]),
        "top1_km2": km(t1["area_km2"]), "top2_km2": km(t2["area_km2"]),
        "forest_wc_pct": pct(pcmp[forest_name]["share_agreeing_with_worldcover"]),
        "forest_es_pct": pct(pcmp[forest_name]["share_agreeing_with_esri"]),
        "crop_wc_pct": pct(pcmp["Cropland and grassland"]["share_agreeing_with_worldcover"]),
        "crop_es_pct": pct(pcmp["Cropland and grassland"]["share_agreeing_with_esri"]),
        "loss_total_km2": km(H["loss_total_km2"]), "tc2000_km2": km(H["treecover2000_ge30_km2"]),
        "loss_max_year": ranked[0][0], "loss_max_km2": km(ranked[0][1]),
        "loss_second_year": ranked[1][0], "loss_second_km2": km(ranked[1][1]),
        "loss_min_year": ranked[-1][0], "loss_min_km2": km(ranked[-1][1]),
        "loss_in_forest_pct": pct(comp["hansen"]["share_of_loss_2001_2024_mapped_forest_2025"]),
        "forest_tc2000_pct": pct(comp["hansen"]["share_mapped_forest_with_treecover2000_ge30"]),
        "fritz_locations": str(ref["fritz2017"]["locations_in_box"]), "lesiv_sites": str(ref["lesiv2025"]["sites_in_box"]),
        "target_points": str(ref["target_points"]),
        "lesiv_match": str(les["dominant_class_matches"]),
        "ua_assumed": ", ".join(f"{v:.2f} for {k.lower()}" for k, v in design["expected_users_accuracy"].items()),
        "target_se": f"{design['target_overall_se']}", "n_formula": str(design["formula_sample_size"]),
        "n_formula_01": f"{design['formula_sample_size_at_se_0_01']:,}", "min_per_class": str(design["min_per_class"]),
        "n_sample": str(design["total_sample_size"]),
        "allocation": ", ".join(f"{v} {k.lower()}" for k, v in design["allocation"].items()),
    }
    vals.update(accuracy_blocks(acc, design))
    fml_s = tr["fml_share_of_box"]
    ua_b = design["expected_users_accuracy"]["Built-up"]
    need(side["with_neither"] < side["with_worldcover"], "the map rarely sides with neither where the maps disagree")
    need(M["Built-up"]["area_km2"] > A["Built-up"]["worldcover2021"], "more built-up land than WorldCover")
    vals.update({
        "title": title(acc),
        "plant_share_pct": f"{(fml_s['31'] + fml_s['32']) * 100:.2f}",
        "built_prec_rw": p2(built_rw),
        "side_neither_pct": pct(side["with_neither"]),
        "train_forest_pct": pct((s["Forest"] + s["Plantation forest"]) / n_train),
        "train_built_pct": pct(s["Built-up"] / n_train),
        "built_pct": f"{M['Built-up']['share'] * 100:.1f}",
        "ua_built": p2(ua_b), "n_built_sample": str(design["allocation"]["Built-up"]),
        "ua_built_hw": p2(design["expected_users_ci_halfwidth"]["Built-up"]),
    })
    return vals




def fpc_built_precision(fin, rf):
    return rf(fin, "block_10km")["per_class"]["Built-up"]["precision"]


def figures():
    from PIL import Image
    fig = ROOT / "paper" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for src, dst in (("m01_composite", "fig1_composite"), ("m02_landcover", "fig2_landcover"),
                     ("m03_agreement", "fig3_agreement"), ("m04_hansen_loss_years", "fig4_hansen"),
                     ("m05_samples", "fig_samples"), ("m06_cv", "fig_cv"), ("m07_areas", "fig_areas")):
        im = Image.open(study.DOCS / "assets" / "maps" / f"{src}.webp").convert("RGB")
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
        im.save(fig / f"{dst}.jpg", quality=88, optimize=True)


def main():
    vals = build()
    tpl = (ROOT / "paper" / "template.md").read_text(encoding="utf-8")
    keys = set(re.findall(r"\{\{(\w+)\}\}", tpl))
    missing = sorted(keys - set(vals))
    if missing:
        raise SystemExit(f"template keys without values: {missing}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], tpl)
    assert "{{" not in out, "unfilled placeholder"
    figures()
    (ROOT / "paper" / "paper.md").write_text(out, encoding="utf-8")
    print("wrote paper/paper.md with", len(keys), "values")


if __name__ == "__main__":
    main()
