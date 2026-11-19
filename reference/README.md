# Reference labels

Put the CSV exported from `docs/label.html` here as `labels.csv` (columns `id, lon, lat, label, note, interpreter,
labelled_at`), then run

```bash
python scripts/s09_accuracy.py
python scripts/build_paper.py && pandoc paper/paper.md -o paper/paper.docx --resource-path=. && python scripts/build_page.py
pytest
```

Until every point has a label, `outputs/accuracy.json` stays `pending` and the paper and page say so.
