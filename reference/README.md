# Reference labels

Put the CSV exported from `docs/label.html` here as `labels.csv` (columns `id, lon, lat, label, note, interpreter,
labelled_at`), then run

```bash
python scripts/s09_accuracy.py
python scripts/build_paper.py && pandoc paper/paper.md -o paper/paper.docx --resource-path=. && python scripts/build_page.py
pytest
```

Until every point has a label, `outputs/accuracy.json` stays `pending` and the paper and page say so.

## Labelling protocol

The class definitions follow the translation of WorldCover and Esri classes used in the study (paper, Section 2.5).

- **Forest (natural and plantation)**: Tree crowns cover the largest part of the yellow square: natural forest, tree plantations (pine, cypress, eucalyptus) and woodlots. Scattered trees over fields, covering less of the square than the crops, are cropland.
- **Cropland and grassland**: Fields, fallow, pasture, grassland and shrubland, and tea. Tea counts here even though it is a woody crop.
- **Built-up**: Buildings, roads, paved yards and other sealed surfaces cover the largest part of the square.
- **Water and wetland**: Open water, or flooded or wetland vegetation (reeds, swamp), as in WorldCover's herbaceous wetland and the Esri map's flooded vegetation.
- **Bare or other**: Bare soil, rock or sand with no crop pattern. A ploughed field with clear field edges is cropland.
- **Unclear**: The imagery is cloudy, blurred or too coarse, the square seems to have changed and the 2025 state cannot be judged, or two classes cover it equally.

Mixed squares: choose the class that covers the largest part of the yellow square. Imagery date: the basemap date is not shown and may be several years old; label what it shows, unless it shows clear signs of recent change (fresh clearing, construction), in which case choose Unclear.

The interpreter is the map author and has seen the map, so labelling is blind per point, not fully independent.
