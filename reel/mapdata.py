#!/usr/bin/env python3
"""Download terrain + satellite imagery for the 3D map intro and resample both
onto one regular lat/lon grid.

Sources:
  * elevation: AWS Terrain Tiles (Terrarium encoding), open data
  * imagery:   Sentinel-2 cloudless 2021 by EOX IT Services GmbH (s2maps.eu),
               contains modified Copernicus Sentinel data 2021 — CC BY-NC-SA 4.0

Usage: python3 mapdata.py <out_dir>
"""
import concurrent.futures as cf
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "map"
os.makedirs(OUT, exist_ok=True)

LON0, LON1 = 43.9, 45.9
LAT0, LAT1 = 39.6, 41.1
DEM_Z, SAT_Z = 11, 12
GRID = (1400, 1400)      # height samples (rows, cols)
TEX = (4096, 4096)       # texture size (rows, cols)

DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
SAT_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg"


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    lr = math.radians(lat)
    y = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    return x, y


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reel-map/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise


def mosaic(url, z, mode):
    x0, y1 = tile_xy(LON0, LAT0, z)
    x1, y0 = tile_xy(LON1, LAT1, z)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    jobs = {(tx, ty): url.format(z=z, x=tx, y=ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)}
    with cf.ThreadPoolExecutor(16) as ex:
        data = dict(zip(jobs, ex.map(fetch, jobs.values())))
    ch = 3
    img = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, ch), np.uint8)
    for (tx, ty), b in data.items():
        t = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        img[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t
    print(f"{mode}: z{z} {len(jobs)} tiles -> {img.shape}", flush=True)
    return img, tx0, ty0


def resample(img, tx0, ty0, z, shape):
    """Sample a Web-Mercator mosaic on a regular lat/lon grid (row 0 = north)."""
    rows, cols = shape
    lons = np.linspace(LON0, LON1, cols)
    lats = np.linspace(LAT1, LAT0, rows)
    n = 2 ** z
    px = ((lons + 180) / 360 * n - tx0) * 256
    lr = np.radians(lats)
    py = ((1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n - ty0) * 256
    import cv2
    mx, my = np.meshgrid(px.astype(np.float32), py.astype(np.float32))
    return cv2.remap(img, mx - 0.5, my - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


if __name__ == "__main__":
    dem, dx, dy = mosaic(DEM_URL, DEM_Z, "dem")
    d = dem.astype(np.float32)
    elev = d[..., 0] * 256 + d[..., 1] + d[..., 2] / 256 - 32768
    h = resample(elev, dx, dy, DEM_Z, GRID)
    np.save(os.path.join(OUT, "heights.npy"), h.astype(np.float32))
    sat, sx, sy = mosaic(SAT_URL, SAT_Z, "sat")
    tex = resample(sat, sx, sy, SAT_Z, TEX)
    Image.fromarray(tex).save(os.path.join(OUT, "sat.jpg"), quality=93)
    json.dump(dict(lon0=LON0, lon1=LON1, lat0=LAT0, lat1=LAT1, hmin=float(h.min()), hmax=float(h.max())),
              open(os.path.join(OUT, "meta.json"), "w"))
    print("elev range", h.min(), h.max())
