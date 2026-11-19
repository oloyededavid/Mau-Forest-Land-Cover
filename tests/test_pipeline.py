"""Tests for the Mau Forest land-cover study: helper functions on small arrays, the committed outputs, the area
estimate maths, and that the paper and page have every placeholder filled. None of these need the rasters in
data/, which are not committed."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper  # noqa: E402
import features  # noqa: E402
import s03_samples  # noqa: E402
import s04_cv  # noqa: E402
import s07_compare  # noqa: E402
import s09_accuracy  # noqa: E402
import study  # noqa: E402
from spatial_stats_core import (SpatialBlockKFold, allocate_stratified_sample,  # noqa: E402
                                assign_spatial_blocks, olofsson_area_estimate)

OUT = ROOT / "outputs"
DOCS = ROOT / "docs"


def jload(name):
    return json.loads((OUT / name).read_text())


# ---------------------------------------------------------------------------------------------- class scheme

def test_candidate_classes_do_not_share_source_codes_except_the_two_tree_classes():
    seen = {}
    for code, c in study.CANDIDATE_CLASSES.items():
        for key in ("worldcover", "esri"):
            for v in c[key]:
                seen.setdefault((key, v), []).append(c["name"])
    shared = {k: v for k, v in seen.items() if len(v) > 1}
    assert set(map(tuple, shared.values())) == {("Forest", "Plantation forest")}


def test_final_scheme_is_a_subset_of_the_candidates_and_records_its_decisions():
    cl = jload("classes.json")
    codes = [c["code"] for c in cl["final"]]
    assert set(codes) <= set(study.CANDIDATE_CLASSES)
    assert len(set(c["name"] for c in cl["final"])) == len(codes)
    for c in cl["final"]:
        assert re.fullmatch(r"#[0-9a-f]{6}", c["colour"])
    d = cl["plantation_decision"]
    kept = d["precision"] >= d["threshold"] and d["recall"] >= d["threshold"]
    assert kept == d["kept"]
    assert (2 in codes) == d["kept"]


def test_agreement_labels_need_both_maps_and_split_trees_by_the_management_map():
    wc = np.array([[10, 10, 10, 40, 50, 80, 10]])
    es = np.array([[2, 2, 2, 5, 7, 1, 5]])
    fml = np.array([[11, 32, 53, 0, 0, 0, 20]])
    loss = np.zeros_like(wc)
    lab = s03_samples.agreement_labels(wc, es, fml, loss)
    assert lab.tolist() == [[1, 2, 0, 3, 4, 5, 0]]
    loss[0, 0] = 23   # recent Hansen loss removes a tree label
    assert s03_samples.agreement_labels(wc, es, fml, loss)[0, 0] == 0


def test_homogeneous_keeps_only_uniform_neighbourhoods():
    lab = np.zeros((7, 7), dtype=np.uint8)
    lab[1:6, 1:6] = 3
    lab[3, 5] = 4
    out = s03_samples.homogeneous(lab)
    assert out[2, 2] == 3 and out[3, 3] == 3
    assert out[3, 4] == 0 and out[3, 5] == 0          # next to the odd cell
    assert out[1, 1] == 0                              # edge of the patch touches unlabelled cells


def test_stratified_sample_caps_each_class():
    lab = np.zeros((50, 50), dtype=np.uint8)
    lab[:10] = 1
    lab[10:12] = 4
    r, c, y = s03_samples.stratified_sample(lab, 30, np.random.default_rng(0))
    assert (y == 1).sum() == 30 and (y == 4).sum() == 30
    assert np.all(lab[r, c] == y)
    lab[10:12] = 0; lab[10, 0] = 4
    _, _, y = s03_samples.stratified_sample(lab, 30, np.random.default_rng(0))
    assert (y == 4).sum() == 1


def test_indices_on_known_reflectance():
    b = {k: np.array([v]) for k, v in dict(B03=0.1, B04=0.05, B08=0.45, B8A=0.4, B11=0.2, B12=0.1).items()}
    i = features.indices(b)
    assert i["NDVI"][0] == pytest.approx(0.8)
    assert i["NDWI"][0] == pytest.approx(-0.636363, abs=1e-5)
    assert i["NDMI"][0] == pytest.approx(1 / 3)
    assert i["NBR"][0] == pytest.approx(0.6)


def test_crosswalk_covers_every_code_in_the_box():
    # codes seen in the box in WorldCover 2021 and Esri 2023 (outputs of the comparison step)
    for key, codes in (("worldcover2021", [10, 20, 30, 40, 50, 60, 80, 90]), ("esri2023", [1, 2, 4, 5, 7, 8, 9, 10, 11])):
        assert set(codes) <= set(s07_compare.CROSSWALK[key])


# --------------------------------------------------------------------------------------------- blocked CV

def test_blocked_folds_keep_whole_blocks_on_one_side():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 60_000, size=(800, 2))
    y = rng.integers(1, 4, 800)
    X = rng.normal(size=(800, 3))
    blocks = assign_spatial_blocks(xy, 10_000.0)
    sp = s04_cv.splits("block_10km", X, y, xy)
    assert len(sp) == s04_cv.N_SPLITS
    tested = np.zeros(len(y), dtype=int)
    for tr, te in sp:
        assert not set(blocks[tr]) & set(blocks[te])
        tested[te] += 1
    assert np.all(tested == 1)


def test_random_folds_mix_blocks():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 60_000, size=(800, 2))
    y = rng.integers(1, 4, 800)
    blocks = assign_spatial_blocks(xy, 10_000.0)
    tr, te = s04_cv.splits("random", np.zeros((800, 1)), y, xy)[0]
    assert set(blocks[tr]) & set(blocks[te])


def test_cv_output_has_every_scheme_and_blocked_test_cells_are_further_away():
    cv = jload("cv.json")
    for scheme in ("candidate_scheme", "final_scheme"):
        r = cv[scheme]["results"]
        assert set(r) == {"random", "block_5km", "block_10km", "block_20km"}
        for v in r.values():
            assert sum(v["test_sizes"]) == cv["n_samples"]
            for m in v["models"].values():
                cm = np.array(m["confusion_matrix"])
                assert cm.sum() == cv["n_samples"]
                assert m["overall_accuracy"] == pytest.approx(np.trace(cm) / cm.sum())
        med = {k: v["fold_distances_m"]["mean_fold_median_distance"] for k, v in r.items()}
        assert med["random"] < med["block_5km"] < med["block_10km"] < med["block_20km"]


# ------------------------------------------------------------------------------------- area estimate maths

def olofsson_by_hand(n, area):
    """Olofsson et al. (2014) eq. 9-10 and 4-5, written out with loops."""
    q = len(area)
    W = [a / sum(area) for a in area]
    ni = [sum(row) for row in n]
    p = [[W[i] * n[i][j] / ni[i] for j in range(q)] for i in range(q)]
    pk = [sum(p[i][k] for i in range(q)) for k in range(q)]
    se = [sum(W[i] ** 2 * (n[i][k] / ni[i]) * (1 - n[i][k] / ni[i]) / (ni[i] - 1) for i in range(q)) ** 0.5
          for k in range(q)]
    oa = sum(p[i][i] for i in range(q))
    ua = [n[i][i] / ni[i] for i in range(q)]
    pa = [p[j][j] / pk[j] for j in range(q)]
    z = 1.959963984540054
    oa_hw = z * sum(W[i] ** 2 * ua[i] * (1 - ua[i]) / (ni[i] - 1) for i in range(q)) ** 0.5   # eq. 5
    ua_hw = [z * (ua[i] * (1 - ua[i]) / (ni[i] - 1)) ** 0.5 for i in range(q)]                # eq. 6
    area_hw = [z * se[k] * sum(area) for k in range(q)]                                       # eq. 10-11
    pa_hw = []                                                                                 # eq. 7
    for j in range(q):
        nj = sum(W[i] / ni[i] * n[i][j] for i in range(q))                                    # estimated N.j / N
        a = (W[j] ** 2) * ((1 - pa[j]) ** 2) * ua[j] * (1 - ua[j]) / (ni[j] - 1)
        b = (pa[j] ** 2) * sum(W[i] ** 2 * (n[i][j] / ni[i]) * (1 - n[i][j] / ni[i]) / (ni[i] - 1)
                               for i in range(q) if i != j)
        pa_hw.append(z * ((a + b) / nj ** 2) ** 0.5)
    return pk, se, oa, ua, pa, oa_hw, ua_hw, area_hw, pa_hw


def test_olofsson_matches_a_hand_calculation():
    n = [[97, 2, 1], [3, 279, 18], [0, 1, 49]]
    area = [22353.0, 1122543.0, 610228.0]
    r = olofsson_area_estimate(n, area)
    pk, se, oa, ua, pa, oa_hw, ua_hw, area_hw, pa_hw = olofsson_by_hand(n, area)
    t = r.class_table
    assert r.overall_accuracy == pytest.approx(oa)
    lo, hi = r.overall_accuracy_ci
    assert (hi - lo) / 2 == pytest.approx(oa_hw)
    assert np.allclose(t["users_ci_halfwidth"], ua_hw)
    assert np.allclose(t["area_ci_halfwidth"], area_hw)
    assert np.allclose(t["producers_ci_halfwidth"], pa_hw)
    assert np.allclose(t["estimated_proportion"], pk)
    assert np.allclose(t["proportion_se"], se)
    assert np.allclose(t["users_accuracy"], ua)
    assert np.allclose(t["producers_accuracy"], pa)
    assert t["estimated_area"].sum() == pytest.approx(sum(area))


def test_rectangular_estimator_matches_the_package_when_the_extra_column_is_empty():
    n = np.array([[97, 2, 1, 0], [3, 279, 18, 0], [0, 1, 49, 0]])
    area = [22353.0, 1122543.0, 610228.0]
    r = s09_accuracy.olofsson_rect(n, area)
    t = olofsson_area_estimate(n[:, :3], area).class_table
    for k in ("estimated_area", "area_ci_halfwidth", "users_accuracy", "users_ci_halfwidth", "producers_accuracy",
              "producers_ci_halfwidth"):
        assert np.allclose(np.asarray(r[k])[:3], t[k]), k
    assert r["estimated_area"][3] == 0
    # one reference-only point lowers user's accuracy of its stratum and gets an area of its own
    n[1, 3] = 1; n[1, 1] -= 1
    r2 = s09_accuracy.olofsson_rect(n, area)
    assert r2["users_accuracy"][1] < r["users_accuracy"][1] and r2["estimated_area"][3] > 0
    assert sum(r2["estimated_area"]) == pytest.approx(sum(area))


def test_sample_size_follows_the_formula():
    d = jload("sample_design.json")
    names = list(d["mapped_share"])
    W = np.array([d["mapped_share"][k] for k in names])
    U = np.array([d["expected_users_accuracy"][k] for k in names])
    n = int(np.ceil((np.sum(W * np.sqrt(U * (1 - U))) / d["target_overall_se"]) ** 2))
    assert n == d["formula_sample_size"]
    a = allocate_stratified_sample(W, U, d["target_overall_se"], d["min_per_class"])
    assert [int(v) for v in a["recommended_allocation"]] == [d["allocation"][k] for k in names]


def test_accuracy_script_is_pending_until_every_point_has_a_label_then_estimates():
    sample = pd.read_csv(OUT / "reference_sample.csv")
    names = [c["name"] for c in jload("classes.json")["final"]]
    mapped = {k: v["area_km2"] for k, v in jload("map_areas.json")["mapped"].items()}
    part = pd.DataFrame({"id": sample["id"][:10], "label": sample["map_class"][:10]})
    assert s09_accuracy.estimate(sample, part, names, mapped)["status"] == "pending"
    # a fully labelled sample in which the map is always right gives accuracy 1 and areas equal to the mapped areas
    full = pd.DataFrame({"id": sample["id"], "label": sample["map_class"]})
    full.loc[0, "label"] = "Unclear"
    r = s09_accuracy.estimate(sample, full, names, mapped)
    assert r["status"] == "complete" and r["n_unclear"] == 1 and r["n_used"] == len(sample) - 1
    assert r["overall_accuracy"] == pytest.approx(1.0)
    for k in names:
        assert r["per_class"][k]["estimated_area"] == pytest.approx(mapped[k])
    # "Bare or other" is a reference-only column: an error for its map class, never merged into a map class
    full.loc[1, "label"] = "Bare or other"
    r = s09_accuracy.estimate(sample, full, names, mapped)
    i = names.index(sample.loc[1, "map_class"])
    assert r["n_bare_or_other"] == 1 and r["reference_classes"][-1] == "Bare or other"
    assert r["error_matrix"][i][-1] == 1 and r["overall_accuracy"] < 1
    assert r["bare_or_other"]["estimated_area"] > 0
    # duplicate ids in the labels file are refused
    dup = pd.concat([full, full.iloc[[3]]], ignore_index=True)
    with pytest.raises(SystemExit):
        s09_accuracy.estimate(sample, dup, names, mapped)
    full.loc[2, "label"] = "Something else"
    with pytest.raises(SystemExit):
        s09_accuracy.estimate(sample, full, names, mapped)


# -------------------------------------------------------------------------------------------- data checks

def test_scenes_are_the_dry_season_scenes_over_the_four_tiles():
    s = pd.read_csv(OUT / "scenes.csv")
    meta = jload("composite.json")
    assert len(s) == meta["n_scenes"] and s["scene_id"].is_unique
    assert set(s["mgrs_tile"]) == set(study.TILES)
    assert s["date"].min() >= "2025-01-01" and s["date"].max() <= "2025-03-31"
    assert (s["cloud_cover_pct"] < study.MAX_CLOUD).all()
    assert (s["processing_baseline"] >= 4.0).all()     # the -1000 offset applies to every scene
    assert meta["clear_count_in_box"]["share_zero"] < 0.001


def test_training_sample_is_clean():
    t = pd.read_csv(OUT / "training_samples.csv")
    tr = jload("training.json")
    assert set(t["label"]) <= set(study.CANDIDATE_CLASSES)
    assert t[study.FEATURES].notna().all().all()
    assert t["B04"].between(-0.05, 1.5).all() and t["NDVI"].between(-1, 1).all()
    assert t["lon"].between(study.BBOX[0], study.BBOX[2]).all() and t["lat"].between(study.BBOX[1], study.BBOX[3]).all()
    for name, n in tr["sampled"].items():
        code = next(k for k, v in study.CANDIDATE_CLASSES.items() if v["name"] == name)
        assert (t["label"] == code).sum() == n <= tr["n_per_class_max"]
    assert not t.duplicated(["row", "col"]).any()


def test_mapped_areas_add_up():
    a = jload("map_areas.json")
    shares = sum(v["share"] for v in a["mapped"].values())
    assert shares == pytest.approx(1.0)
    assert sum(v["cells"] for v in a["mapped"].values()) == a["classified_cells"]
    assert a["unclassified_cells_in_box"] / a["box_cells"] < 0.001
    names = [c["name"] for c in jload("classes.json")["final"]]
    assert list(a["mapped"]) == names


def test_reference_sample_matches_the_design_and_the_tool_is_blind():
    s = pd.read_csv(OUT / "reference_sample.csv")
    d = jload("sample_design.json")
    assert len(s) == d["total_sample_size"] and s["id"].is_unique
    assert s["map_class"].value_counts().to_dict() == d["allocation"]
    assert s["lon"].between(study.BBOX[0], study.BBOX[2]).all() and s["lat"].between(study.BBOX[1], study.BBOX[3]).all()
    tool = json.loads((DOCS / "data" / "label_points.json").read_text())
    assert [p["id"] for p in tool["points"]] == list(s["id"])
    assert all(set(p) == {"id", "lon", "lat"} for p in tool["points"])
    assert tool["classes"] == d["label_classes"]
    names = [c["name"] for c in jload("classes.json")["final"]]
    assert tool["classes"][:len(names)] == names and tool["classes"][-2:] == [s09_accuracy.OTHER, s09_accuracy.UNCLEAR]


def test_comparison_and_hansen_outputs_are_consistent():
    c = jload("comparison.json")
    assert c["agree_with_both"] <= min(c["agree_with_worldcover"], c["agree_with_esri"])
    assert c["agree_with_both"] + c["agree_with_neither"] <= 1
    tot = {k: sum(v[k] for v in c["areas_km2"].values()) for k in ("this_map", "worldcover2021", "esri2023")}
    assert tot["worldcover2021"] == pytest.approx(tot["esri2023"]) == pytest.approx(c["compared_km2"])
    h = jload("hansen_loss_by_year.json")
    assert list(h["loss_km2_by_year"]) == [str(y) for y in range(2001, 2025)]
    assert sum(h["loss_km2_by_year"].values()) == pytest.approx(h["loss_total_km2"])
    assert h["loss_total_km2"] < h["treecover2000_ge30_km2"] < h["box_land_km2"]
    r = jload("reference_search.json")
    assert r["enough_for_accuracy"] is False
    assert c["lesiv2015_sites"]["n_sites"] == r["lesiv2025"]["sites_in_box"]


def test_accuracy_output_is_pending_or_complete_and_never_invented():
    a = jload("accuracy.json")
    assert a["status"] in ("pending", "complete")
    labels = ROOT / "reference" / "labels.csv"
    if a["status"] == "complete":
        assert labels.exists()
    else:
        assert "overall_accuracy" not in a and "per_class" not in a


# ------------------------------------------------------------------------------------- paper and page

def test_map_classes_are_not_published_on_the_site_while_labelling_is_pending():
    if jload("accuracy.json")["status"] == "pending":
        assert not (DOCS / "data" / "reference_sample.csv").exists()
        assert "reference_sample.csv\" download" not in (DOCS / "index.html").read_text()


def test_title_claims_area_estimates_only_when_they_exist():
    status = jload("accuracy.json")["status"]
    expected = build_paper.TITLE["complete" if status == "complete" else "pending"]
    paper_title = re.search(r'^title: "(.*)"$', (ROOT / "paper" / "paper.md").read_text(), re.M).group(1)
    assert paper_title == expected
    html = (DOCS / "index.html").read_text()
    assert f"<title>{expected}</title>" in html and f"<i>{expected}</i>" in html
    cff = re.search(r'^title: "(.*)"$', (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    assert cff == expected, "CITATION.cff title must match the paper title"
    if status != "complete":
        assert "stratified area estimates for 2025" not in html


def test_no_visible_placeholder_text_on_the_page():
    html = (DOCS / "index.html").read_text()
    visible = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "placeholder" not in visible.lower()
    assert "<!-- draft: author to revise -->" in html


def test_every_template_placeholder_has_a_value():
    tpl = (ROOT / "paper" / "template.md").read_text()
    keys = set(re.findall(r"\{\{(\w+)\}\}", tpl))
    vals = build_paper.build()
    assert keys <= set(vals), sorted(keys - set(vals))


def test_built_paper_and_page_have_no_placeholders_and_match_the_build():
    vals = build_paper.build()
    tpl = (ROOT / "paper" / "template.md").read_text()
    vals["date"] = re.search(r'^date: "(.*)"$', (ROOT / "paper" / "paper.md").read_text(), re.M).group(1)
    expected = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], tpl)
    assert (ROOT / "paper" / "paper.md").read_text() == expected, "paper.md is out of date: run build_paper.py"
    html = (DOCS / "index.html").read_text()
    assert not re.search(r"\{\{\s*\w+\s*\}\}", html)
    if json.loads((OUT / "accuracy.json").read_text())["status"] == "pending":
        assert "Pending the reference sample" in html


def test_page_links_and_assets_exist():
    for page in ("index.html", "label.html"):
        html = (DOCS / page).read_text()
        for ref in re.findall(r'(?:src|href)="([^"#:]+)"', html):
            if ref.startswith(("http", "data:", "mailto")) or ref in ("./",):
                continue
            assert (DOCS / ref).exists(), f"{page}: {ref} missing"
        ids = set(re.findall(r'id="([^"]+)"', html))
        for a in re.findall(r'href="#([^"]+)"', html):
            assert a in ids, f"{page}: anchor #{a} has no target"
    ov = json.loads((DOCS / "data" / "overlays.json").read_text())
    for layer in ov["layers"]:
        assert (DOCS / layer["file"]).exists()


def test_body_counter_namespace_and_no_inflated_words():
    html = (DOCS / "index.html").read_text()
    assert 'data-counter="mau-forest-land-cover"' in html
    text = " ".join(p.read_text() for p in [DOCS / "index.html", DOCS / "label.html", ROOT / "README.md"])
    for w in ("novel", "groundbreaking", "first study", "cutting-edge"):
        assert w not in text, w
