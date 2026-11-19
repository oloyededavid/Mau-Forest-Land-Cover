# Mapping land cover in the Mau Forest Complex, Kenya, with Sentinel-2

**Author:** David Oloyede

**Study page:** <https://oloyededavid.github.io/Mau-Forest-Land-Cover/>

A land-cover map of a box around the Mau Forest Complex for 2025, made with a random forest on a Sentinel-2 dry-season composite, tested with spatially blocked cross-validation, and compared with ESA WorldCover 2021, the Esri / Impact Observatory land cover map for 2023 and Hansen forest loss. The research note is [`paper/paper.md`](paper/paper.md) (Word: `paper/paper.docx`); the page is built from it. Every number in both is read from the files in [`outputs/`](outputs/) that the scripts write; `scripts/build_paper.py` stops if a sentence in `paper/template.md` no longer matches them.

## What this study can and cannot say

- The study area is a rectangle (35.30–36.10°E, 0.15–0.95°S), **not the legal boundary** of the forest reserves.
- **Training labels come from existing maps**: cells where WorldCover 2021 and Esri 2023 agree, in uniform 3 × 3 neighbourhoods, with tree cover split by a 2015 forest management map (Lesiv et al., 2022). No labels were drawn by hand and there are no field data.
- Cross-validated accuracy is agreement with those labels on easy cells. **It is not map accuracy.**
- A plantation class was tried and failed a rule fixed in advance under 10 km blocked cross-validation, so plantations (and some tea) are mapped as forest. No bare class could be trained.
- No open reference data set had enough independent points inside the box (`outputs/reference_search.json`). **Accuracy and stratified area estimates are pending** a labelled stratified sample. The study ships the sample (`outputs/reference_sample.csv`, which holds each point's map class, so do not open it before labelling; the labelling tool reads `docs/data/label_points.json`, which has locations only) and a labelling tool (`docs/label.html`). Until the labels are added, the paper and page say "pending the reference sample" and report no such numbers.
- The composite covers one dry season (January–March 2025). Hansen loss is a separate product and is used as context only.

## Data

| Layer | Source | Access |
| :--- | :--- | :--- |
| Surface reflectance | Sentinel-2 L2A, January–March 2025, cloud cover under 20% | Microsoft Planetary Computer (`sentinel-2-l2a`) |
| Elevation, slope | Copernicus DEM GLO-30 | Planetary Computer (`cop-dem-glo-30`) |
| Labels and comparison | ESA WorldCover 10 m 2021 v200 (Zanaga et al., 2022) | Planetary Computer (`esa-worldcover`) |
| Labels and comparison | Esri / Impact Observatory annual land cover 2023 (Karra et al., 2021) | Planetary Computer (`io-lulc-annual-v02`) |
| Plantation labels | Global forest management map 2015 (Lesiv et al., 2022, *Scientific Data*; data on Zenodo, 2021, file v3.2, doi:10.5281/zenodo.5879022) | Zenodo |
| Forest loss context | Hansen Global Forest Change v1.12, tile 00N_030E (Hansen et al., 2013) | Google Cloud Storage |
| Reference search | Geo-Wiki 2011–2012 (Fritz et al., 2016, doi:10.1594/PANGAEA.869682); Geo-Wiki 2015 at 10 m (Lesiv et al., 2025, doi:10.5281/zenodo.14871660) | Zenodo |

## Reproduce

```bash
pip install -r requirements-dev.txt     # includes spatial-stats-core from GitHub
python scripts/s01_composite.py         # Sentinel-2 median composite, about 15 min, 15 GB RAM machine
python scripts/s02_layers.py            # DEM, WorldCover, Esri, forest management map, Hansen on the 20 m grid
python scripts/s03_samples.py           # map-agreement training sample
python scripts/s04_cv.py                # random and blocked cross-validation, class decision
python scripts/s05_classify.py          # random forest map, about 20 min
python scripts/s06_reference_search.py  # count open reference points in the box (streams about 2 GB once)
python scripts/s07_compare.py           # comparison with WorldCover, Esri, Hansen and the 2015 sites
python scripts/s08_sample_design.py     # stratified reference sample and the labelling tool's points
python scripts/s09_accuracy.py          # accuracy and area estimates once reference/labels.csv exists
python scripts/s10_figures.py           # maps, chart and interactive-map overlays
python scripts/build_paper.py && pandoc paper/paper.md -o paper/paper.docx --resource-path=. && python scripts/build_page.py
pytest
```

Downloads and rasters go to `data/`, which is not committed. The files in `outputs/` are committed, so the paper, page and tests can be rebuilt without downloading anything.

## Labelling the reference sample

Open `docs/label.html` (or the page online), label every point, export the CSV, save it as `reference/labels.csv`, and run `python scripts/s09_accuracy.py`, then rebuild the paper and page. See [`reference/README.md`](reference/README.md).

## Repository layout

```
scripts/     pipeline (s01 to s10), paper and page build, page template and page end
outputs/     result files (JSON, CSV) read by the paper, page and tests
paper/       template.md with placeholders, built paper.md and paper.docx, figures
docs/        study page (GitHub Pages): index.html, label.html, assets/, data/, vendor/leaflet
reference/   where the exported reference labels go
tests/
```

## Licence

Code: MIT (`LICENSE`). Derived data and figures: CC BY 4.0 (`LICENSE-DATA.md`). Source datasets keep their own terms.
