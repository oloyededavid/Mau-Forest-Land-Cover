"""Feature stack for classification: Sentinel-2 median reflectance, four indices and terrain.

Indices (reflectance bands):
    NDVI = (B08 - B04) / (B08 + B04)          vegetation
    NDWI = (B03 - B08) / (B03 + B08)          open water (McFeeters 1996)
    NDMI = (B8A - B11) / (B8A + B11)          canopy and soil moisture
    NBR  = (B8A - B12) / (B8A + B12)          normalised burn ratio type index
"""
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def indices(bands):
    """bands: dict of reflectance arrays (float). Returns dict with NDVI, NDWI, NDMI, NBR."""
    return {
        "NDVI": nd(bands["B08"], bands["B04"]),
        "NDWI": nd(bands["B03"], bands["B08"]),
        "NDMI": nd(bands["B8A"], bands["B11"]),
        "NBR": nd(bands["B8A"], bands["B12"]),
    }


def read_stack(window=None):
    """Return (features array [n_features, rows, cols] float32, valid mask, clear_count)."""
    with rasterio.open(study.DATA / "composite.tif") as src:
        raw = src.read(window=window)
        names = list(src.descriptions)
    bands = {}
    valid = np.ones(raw.shape[1:], dtype=bool)
    for i, n in enumerate(names):
        if n in study.S2_BANDS:
            a = raw[i].astype(np.float32)
            valid &= a != -32768
            bands[n] = a / 10000.0
    clear = raw[names.index("clear_count")]
    idx = indices(bands)
    with rasterio.open(study.DATA / "layers.tif") as src:
        lnames = list(src.descriptions)
        elev = src.read(lnames.index("elevation") + 1, window=window)
        slope = src.read(lnames.index("slope") + 1, window=window)
    valid &= elev > -1000
    feats = {**bands, **idx, "elevation": elev, "slope": slope}
    X = np.stack([feats[f].astype(np.float32) for f in study.FEATURES])
    valid &= np.all(np.isfinite(X), axis=0)
    return X, valid, clear
