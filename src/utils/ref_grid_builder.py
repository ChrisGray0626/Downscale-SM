#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Reference Grid (EASE-Grid 2.0) Builder
@Author Chris
@Date 2025/12/12
"""

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from constants import RANGE, RESOLUTION_1KM, RESOLUTION_36KM, REF_GRID_1KM_PATH, REF_GRID_36KM_PATH
from utils.tiff_util import write_tiff


def build_ref_grid(resolution: str):
    if resolution == RESOLUTION_1KM:
        resolution_num = 1000
        dst_path = REF_GRID_1KM_PATH
    else:
        resolution_num = 36000
        dst_path = REF_GRID_36KM_PATH

    lon_min, lat_min, lon_max, lat_max = RANGE

    wgs84_crs = CRS.from_epsg(4326)
    ease_crs = CRS.from_epsg(6933)
    transformer = Transformer.from_crs(wgs84_crs, ease_crs, always_xy=True)

    x_min, y_max = transformer.transform(lon_min, lat_max)
    x_max, y_min = transformer.transform(lon_max, lat_min)

    x_min_aligned = np.floor(x_min / resolution_num) * resolution_num
    x_max_aligned = np.ceil(x_max / resolution_num) * resolution_num
    y_min_aligned = np.floor(y_min / resolution_num) * resolution_num
    y_max_aligned = np.ceil(y_max / resolution_num) * resolution_num

    x_coords = np.arange(x_min_aligned, x_max_aligned + resolution_num, resolution_num)
    y_coords = np.arange(y_max_aligned, y_min_aligned - resolution_num, -resolution_num)
    nrows, ncols = len(y_coords), len(x_coords)

    fill_values = np.full((nrows, ncols), np.nan, dtype=np.float32)
    transform = from_origin(x_min_aligned, y_max_aligned, resolution_num, resolution_num)

    write_tiff(fill_values, dst_path, transform=transform, crs=ease_crs, nodata=np.nan)


def main():
    build_ref_grid(RESOLUTION_1KM)
    build_ref_grid(RESOLUTION_36KM)


if __name__ == "__main__":
    main()
