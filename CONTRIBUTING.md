# Contributing and working on this repository

Static research site (GitHub Pages, `docs/`) on land cover in the Mau Forest Complex, Kenya, in 2025, with the
Python scripts that produce every number and map on it.

## Rules

- Every number in `paper/paper.md` and `docs/index.html` is read from a file in `outputs/` that a script in
  `scripts/` writes. Never type a result into the template, HTML or JavaScript. `scripts/build_paper.py` fills
  `paper/template.md` and stops if a statement in it no longer matches the outputs (`need(...)` checks);
  `scripts/build_page.py` builds the page from the paper.
- Map images in `docs/assets/maps/` and overlays in `docs/data/overlays/` are written by `scripts/s10_figures.py`
  and `scripts/s11_charts.py`.
  Do not edit them by hand.
- Training labels come from existing maps; cross-validated accuracy is agreement with those labels, not map
  accuracy. Say so wherever it is reported.
- Accuracy and area estimates come only from `reference/labels.csv` through `scripts/s09_accuracy.py`. Never fill
  them in by hand; while labels are missing the page says "pending the reference sample".
- The labelling tool must stay blind: `docs/data/label_points.json` holds ids and locations only.
- Design: light only, Satoshi and Zodiak fonts, as on the other study sites. No emojis or decorative icons.

## Pipeline

| Step | Command | Writes |
| :--- | :--- | :--- |
| Composite | `python scripts/s01_composite.py` | `data/composite.tif`, `outputs/scenes.csv`, `outputs/composite.json` |
| Other layers | `python scripts/s02_layers.py` | `data/layers.tif`, `outputs/hansen_loss_by_year.json` |
| Training sample | `python scripts/s03_samples.py` | `outputs/training_samples.csv`, `outputs/training.json` |
| Cross-validation, class decision | `python scripts/s04_cv.py` | `outputs/cv.json`, `outputs/classes.json` |
| Classification | `python scripts/s05_classify.py` | `data/landcover_2025.tif`, `outputs/map_areas.json` |
| Reference search | `python scripts/s06_reference_search.py` | `outputs/reference_search.json`, `outputs/reference_lesiv2015_box.csv` |
| Comparison | `python scripts/s07_compare.py` | `outputs/comparison.json`, `data/agreement.tif` |
| Sample design | `python scripts/s08_sample_design.py` | `outputs/reference_sample.csv`, `outputs/sample_design.json`, `docs/data/label_points.json` |
| Accuracy | `python scripts/s09_accuracy.py` | `outputs/accuracy.json` |
| Figures | `python scripts/s10_figures.py` | `docs/assets/maps/*.webp`, `docs/data/overlays*` |
| Charts from committed outputs (no rasters needed) | `python scripts/s11_charts.py` | `docs/assets/maps/m05_samples*`, `m06_cv*`, `m07_areas*` |
| Paper and page | `python scripts/build_paper.py && pandoc paper/paper.md -o paper/paper.docx --resource-path=. && python scripts/build_page.py` | `paper/paper.md`, `paper/paper.docx`, `docs/index.html`, `docs/data/results.json` |
| Tests | `pytest` | |

The cross-site links in `docs/assets/study-bar.html`, `study-nav.html` and `study-others.html` are kept by the
portfolio's `tools/sync_nav.py`; `build_page.py` inserts them.
