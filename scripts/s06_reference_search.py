"""Count independently collected land-cover reference points inside the study box in two open datasets, and keep
the points of the larger one for a descriptive check.

1. Fritz et al. (2017), Geo-Wiki crowdsourced land cover and land use reference data, 2011-2012
   (Zenodo record 6572482, a copy of PANGAEA doi:10.1594/PANGAEA.869682; CC BY 4.0).
2. Lesiv et al. (2025), global reference data set for land cover mapping at 10 m, reference year 2015
   (Zenodo doi:10.5281/zenodo.14871660; CC BY 4.0). Each sample site is a 100 m square labelled as 100 cells of 10 m.

Neither is a probability sample of the 2025 map, and both predate it, so neither can give map accuracy or area
estimates (Olofsson et al., 2014). The study therefore ships a labelling tool (docs/label.html) for a stratified
random sample drawn from the map; see scripts/s07_sample_design.py.

    python scripts/s06_reference_search.py    # streams about 2 GB once; writes outputs/reference_search.json,
                                              # outputs/reference_lesiv2015_box.csv
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

FRITZ = "https://zenodo.org/api/records/6572482/files/Global%20LULC%20reference%20data%20.csv/content"
LESIV = "https://zenodo.org/api/records/14871660/files/final_reference_data.csv/content"


def inside(lon, lat):
    return study.BBOX[0] <= lon <= study.BBOX[2] and study.BBOX[1] <= lat <= study.BBOX[3]


def fritz():
    cache = study.DATA / "fritz2017.csv"
    if not cache.exists():
        urllib.request.urlretrieve(FRITZ, cache)
    d = pd.read_csv(cache, low_memory=False)
    b = d[[inside(x, y) for x, y in zip(d["lon"], d["lat"])]]
    return {"dataset": "Fritz et al. (2017) Geo-Wiki crowdsourced reference data, 2011-2012",
            "doi": "10.1594/PANGAEA.869682", "records_global": int(len(d)), "records_in_box": int(len(b)),
            "locations_in_box": int(b["pixelID"].nunique())}


def lesiv():
    cache = study.OUT / "reference_lesiv2015_box.csv"
    n = 0
    if not cache.exists():
        rows = []
        with urllib.request.urlopen(LESIV) as resp:
            reader = csv.reader(io.TextIOWrapper(resp, encoding="utf-8"))
            header = next(reader)
            ix, iy = header.index("center_x"), header.index("center_y")
            for row in reader:
                n += 1
                try:
                    if inside(float(row[ix]), float(row[iy])):
                        rows.append(row)
                except ValueError:
                    continue
        keep = ["sampleid", "class_id", "class_name", "reference_year", "center_x", "center_y"]
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keep)
            for r in rows:
                w.writerow([r[header.index(k)] for k in keep])
    b = pd.read_csv(cache)
    return {"dataset": "Lesiv et al. (2025) global reference data set for land cover at 10 m, reference year 2015",
            "doi": "10.5281/zenodo.14871660", "records_global": n if n else None,
            "records_in_box": int(len(b)), "sites_in_box": int(b["sampleid"].nunique()),
            "classes_in_box": b["class_name"].value_counts().to_dict()}


def main():
    study.DATA.mkdir(exist_ok=True)
    old = {}
    p = study.OUT / "reference_search.json"
    if p.exists():
        old = json.loads(p.read_text())
    res = {"fritz2017": fritz(), "lesiv2025": lesiv()}
    if res["lesiv2025"]["records_global"] is None and old.get("lesiv2025", {}).get("records_global"):
        res["lesiv2025"]["records_global"] = old["lesiv2025"]["records_global"]
    res["target_points"] = 100
    res["enough_for_accuracy"] = False
    res["reason"] = ("Too few sites inside the box, collected for other years (2011-2012 and 2015) and not drawn from "
                     "the strata of this map, so they cannot give unbiased accuracy or area estimates for 2025.")
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
