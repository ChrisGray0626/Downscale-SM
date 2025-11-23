#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Precipitation Data
  @Author Chris
  @Date 2025/6/5
"""
import glob

import rasterio
import xarray as xr
from rasterio.transform import from_origin
from tqdm import tqdm

from constant import *
from util.tiff_util import resample_tiff
from util.util import is_tgt_date

# 单位：千米
RESOLUTION = 1
DIR_NAME = PRECIPITATION_NAME
INPUT_DIR_PATH = os.path.join(RAW_DIR_PATH, DIR_NAME)
TIFF_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, CONVERTED_DIR_NAME)
OUTPUT_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, f"{RESOLUTION}km")
STANDARD_GRID_PATH = os.path.join(RESULT_PATH, f"Standard_Grid_{RESOLUTION}km{TIFF_SUFFIX}")


def extract(file_path: str):
    ds = xr.open_dataset(file_path)
    precipitations = ds['precipitation'].isel(time=0)
    lons = ds['lon']
    lats = ds['lat']
    # 翻转数据
    precipitations = precipitations.transpose()
    precipitations = precipitations[::-1, :]
    # 日期转换
    date = ds['time'].values[0]
    date = str(date)[:10].replace('-', '')
    return precipitations, lons, lats, date


def write2tiff(precipitation, lon, lat, output_path):
    transform = from_origin(lon.min(), lat.max(), (lon.max() - lon.min()) / len(lon),
                            (lat.max() - lat.min()) / len(lat))
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            width=precipitation.shape[1],
            height=precipitation.shape[0],
            count=1,
            dtype=precipitation.dtype,
            crs='EPSG:4326',
            transform=transform
    ) as dst:
        dst.write(precipitation.values, 1)


def convert2tiff():
    os.makedirs(TIFF_DIR_PATH, exist_ok=True)
    file_paths = glob.glob(os.path.join(INPUT_DIR_PATH, f"*{NETCDF_SUFFIX}"))
    for file_path in tqdm(file_paths):
        precipitations, lons, lats, date = extract(file_path)
        # 过滤非目标日期
        if not is_tgt_date(date):
            continue
        output_file_path = os.path.join(TIFF_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        write2tiff(precipitations, lons, lats, output_file_path)


def resample():
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    file_paths = glob.glob(os.path.join(TIFF_DIR_PATH, f"*{TIFF_SUFFIX}"))
    for file_path in tqdm(file_paths):
        dst_path = os.path.join(OUTPUT_DIR_PATH,
                                os.path.basename(file_path))
        resample_tiff(file_path, STANDARD_GRID_PATH, dst_path)


def main():
    convert2tiff()
    resample()


if __name__ == "__main__":
    main()
