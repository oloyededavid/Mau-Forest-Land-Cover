"""Shared settings for the Mau Forest land-cover study: the study box, the grid, the dates and the class scheme.

Every script imports this module, so the box, grid and classes are defined once.
"""
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"            # downloads and large rasters, not committed
OUT = ROOT / "outputs"          # small result files (JSON, CSV), committed
DOCS = ROOT / "docs"

# Study box in degrees (lon_min, lat_min, lon_max, lat_max). It is a rectangle around the Mau Forest Complex,
# not the legal forest boundary.
BBOX = (35.30, -0.95, 36.10, -0.15)
DATES = "2025-01-01/2025-03-31"   # dry season composite
MAX_CLOUD = 20                    # eo:cloud_cover limit, per cent
TILES = ("36MYD", "36MYE", "36MZD", "36MZE")

# Analysis grid: WGS 84 / UTM zone 36S, 20 m cells, snapped to whole 20 m multiples
CRS = "EPSG:32736"
RES = 20.0


def grid():
    """Return (transform, width, height) of the analysis grid that covers BBOX."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    lon = np.linspace(BBOX[0], BBOX[2], 50)
    lat = np.linspace(BBOX[1], BBOX[3], 50)
    xs, ys = [], []
    for a in lon:
        for b in (BBOX[1], BBOX[3]):
            x, y = t.transform(a, b)
            xs.append(x); ys.append(y)
    for b in lat:
        for a in (BBOX[0], BBOX[2]):
            x, y = t.transform(a, b)
            xs.append(x); ys.append(y)
    x0 = np.floor(min(xs) / RES) * RES
    x1 = np.ceil(max(xs) / RES) * RES
    y0 = np.floor(min(ys) / RES) * RES
    y1 = np.ceil(max(ys) / RES) * RES
    w = int(round((x1 - x0) / RES))
    h = int(round((y1 - y0) / RES))
    return from_origin(x0, y1, RES, RES), w, h


def box_mask(transform, width, height):
    """True for cells whose centre lies inside the longitude/latitude box (the UTM grid is slightly larger)."""
    from pyproj import Transformer
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    cols = np.arange(width) + 0.5
    rows = np.arange(height) + 0.5
    mask = np.zeros((height, width), dtype=bool)
    step = 256
    for r0 in range(0, height, step):
        rr = rows[r0:r0 + step]
        X, Y = np.meshgrid(transform.c + cols * transform.a, transform.f + rr * transform.e)
        lon, lat = t.transform(X, Y)
        mask[r0:r0 + step] = (lon >= BBOX[0]) & (lon <= BBOX[2]) & (lat >= BBOX[1]) & (lat <= BBOX[3])
    return mask


# ---------------------------------------------------------------------------------------------------------------
# Class scheme. Training labels come from cells where two existing maps agree (map-agreement training):
# ESA WorldCover 2021 v200 and the Esri / Impact Observatory annual land cover (io-lulc-annual-v02).
# Codes: WorldCover 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 80 water, 90 wetland, 95 mangrove;
# Esri 1 water, 2 trees, 4 flooded vegetation, 5 crops, 7 built area, 8 bare ground, 11 rangeland.
# The plantation class is decided in scripts/s05_classify.py from what the data separate (see outputs/classes.json).
CANDIDATE_CLASSES = {
    1: {"name": "Forest", "worldcover": [10], "esri": [2], "colour": "#1f6b35"},
    2: {"name": "Plantation forest", "worldcover": [10], "esri": [2], "colour": "#8cbf3f"},
    3: {"name": "Cropland and grassland", "worldcover": [20, 30, 40], "esri": [5, 11], "colour": "#e8d58a"},
    4: {"name": "Built-up", "worldcover": [50], "esri": [7], "colour": "#c0392b"},
    5: {"name": "Water and wetland", "worldcover": [80, 90], "esri": [1, 4], "colour": "#2c7fb8"},
    6: {"name": "Bare", "worldcover": [60], "esri": [8], "colour": "#9a8f85"},
}
NODATA_CLASS = 0

FEATURES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
            "NDVI", "NDWI", "NDMI", "NBR", "elevation", "slope"]
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

SEED = 20250331
