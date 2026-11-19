"""Put the other layers on the 20 m analysis grid: Copernicus DEM (elevation, slope), ESA WorldCover 2021,
Esri / Impact Observatory land cover 2023, the forest management map of Lesiv et al. (2022), and Hansen Global
Forest Change v1.12 (tree cover 2000, loss year, data mask). Also sums Hansen loss by year inside the box on the
native Hansen grid (cell areas from latitude).

    python scripts/s02_layers.py     # writes data/layers.tif, outputs/hansen_loss_by_year.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study  # noqa: E402

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
HANSEN = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_{}_00N_030E.tif"
FML = "https://zenodo.org/api/records/5879022/files/FML_v3-2_with-colorbar.tif/content"
LAYERS = ["elevation", "slope", "worldcover2021", "esri2023", "fml2015", "hansen_treecover2000", "hansen_lossyear",
          "hansen_datamask"]


def catalog():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def warp(paths, transform, width, height, resampling, dtype, nodata=0):
    """Mosaic one or more rasters onto the grid (first source wins)."""
    out = np.full((height, width), nodata, dtype=dtype)
    have = np.zeros((height, width), dtype=bool)
    for p in paths:
        with rasterio.open(p) as src:
            with WarpedVRT(src, crs=study.CRS, transform=transform, width=width, height=height,
                           resampling=resampling, src_nodata=src.nodata, nodata=nodata) as vrt:
                a = vrt.read(1)
        new = (~have) & (a != nodata)
        out[new] = a[new]
        have |= new
    return out


def slope_deg(z, res):
    gy, gx = np.gradient(z.astype(np.float64), res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def hansen_by_year():
    """Loss area by year inside the box on Hansen's own grid; area of each cell from its latitude."""
    with rasterio.open("/vsicurl/" + HANSEN.format("lossyear")) as s:
        w = from_bounds(*study.BBOX, s.transform).round_offsets().round_lengths()
        loss = s.read(1, window=w)
        tr = s.window_transform(w)
    with rasterio.open("/vsicurl/" + HANSEN.format("datamask")) as s:
        dm = s.read(1, window=from_bounds(*study.BBOX, s.transform).round_offsets().round_lengths())
    with rasterio.open("/vsicurl/" + HANSEN.format("treecover2000")) as s:
        tc = s.read(1, window=from_bounds(*study.BBOX, s.transform).round_offsets().round_lengths())
    lat = tr.f + (np.arange(loss.shape[0]) + 0.5) * tr.e
    R = 6371007.2
    cell = (np.radians(abs(tr.a)) * R) * (np.radians(abs(tr.e)) * R) * np.cos(np.radians(lat))   # m2 per row
    area = np.broadcast_to(cell[:, None], loss.shape) / 1e6
    land = dm == 1
    years = {str(2000 + y): float(area[(loss == y) & land].sum()) for y in range(1, 25)}
    return {
        "source": "Hansen et al. (2013) Global Forest Change 2000-2024 v1.12, tile 00N_030E",
        "box_land_km2": float(area[land].sum()),
        "treecover2000_ge30_km2": float(area[land & (tc >= 30)].sum()),
        "loss_km2_by_year": years,
        "loss_total_km2": float(sum(years.values())),
        "note": "Loss is stand-replacement disturbance of tree cover (vegetation taller than 5 m), not deforestation "
                "alone; plantation harvest counts as loss. The product's processing changed over the record, so "
                "year-to-year comparisons need care.",
    }


def main():
    study.DATA.mkdir(exist_ok=True); study.OUT.mkdir(exist_ok=True)
    transform, width, height = study.grid()
    cat = catalog()
    layers = {}

    dem_items = list(cat.search(collections=["cop-dem-glo-30"], bbox=study.BBOX).items())
    dem = warp([i.assets["data"].href for i in dem_items], transform, width, height, Resampling.bilinear,
               "float32", nodata=-9999.0)
    layers["elevation"] = dem
    layers["slope"] = slope_deg(dem, study.RES)
    print("DEM", [i.id for i in dem_items], flush=True)

    wc = [i for i in cat.search(collections=["esa-worldcover"], bbox=study.BBOX).items() if "2021_v200" in i.id]
    layers["worldcover2021"] = warp([i.assets["map"].href for i in wc], transform, width, height, Resampling.mode, "uint8")
    print("WorldCover", [i.id for i in wc], flush=True)

    esri = [i for i in cat.search(collections=["io-lulc-annual-v02"], bbox=study.BBOX).items() if i.id in ("36M-2023", "37M-2023")]
    esri.sort(key=lambda i: i.id)  # zone 36 covers the box up to x = 833970 m; zone 37 fills the eastern edge
    layers["esri2023"] = warp([i.assets["data"].href for i in esri], transform, width, height, Resampling.mode, "uint8")
    print("Esri", [i.id for i in esri], flush=True)

    fml = warp(["/vsicurl/" + FML], transform, width, height, Resampling.nearest, "int16", nodata=-128)
    layers["fml2015"] = np.where(fml < 0, 0, fml).astype(np.uint8)
    print("FML", flush=True)

    for name in ("treecover2000", "lossyear", "datamask"):
        layers["hansen_" + name] = warp(["/vsicurl/" + HANSEN.format(name)], transform, width, height,
                                        Resampling.nearest, "uint8", nodata=255)
        print("Hansen", name, flush=True)

    profile = dict(driver="GTiff", dtype="float32", width=width, height=height, count=len(LAYERS), crs=study.CRS,
                   transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(study.DATA / "layers.tif", "w", **profile) as dst:
        for i, k in enumerate(LAYERS, start=1):
            dst.write(layers[k].astype(np.float32), i)
            dst.set_band_description(i, k)

    (study.OUT / "hansen_loss_by_year.json").write_text(json.dumps(hansen_by_year(), indent=2))
    print("wrote data/layers.tif and outputs/hansen_loss_by_year.json")


if __name__ == "__main__":
    main()
