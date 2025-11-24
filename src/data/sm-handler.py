#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Solid Moisture Data
  @Author Chris
  @Date 2025/4/25
"""
import glob

import h5py as h5
import numpy as np
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

from Constant import *
from util.TiffUtil import resample_tiff
from util.util import is_tgt_date

# Gap Value
GAP_VALUE = -9999
SM_KEY = "Soil_Moisture_Retrieval_Data_AM/soil_moisture"


# 单位：千米
RESOLUTION = 1

DIR_NAME = SM_NAME
INPUT_DIR_PATH = os.path.join(RAW_DIR_PATH, DIR_NAME)
TIFF_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, CONVERTED_DIR_NAME)
OUTPUT_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, f"{RESOLUTION}km")
STANDARD_GRID_PATH = os.path.join(RESULT_PATH, f"Standard_Grid_{RESOLUTION}km{TIFF_SUFFIX}")


def extract(file_path: str):
    date = os.path.basename(file_path)[13:21]
    f = h5.File(file_path, "r")
    sms = f[SM_KEY][:]
    sms[sms == GAP_VALUE] = np.nan

    return sms, date


def write2tiff(sms, dst_path):
    # EPSG:6933 EASE-Grid 2.0 Global 36km
    pixel_size = 36032.22
    cols = 964
    rows = 406
    ulx = -17367530.44
    uly = 7314540.83

    transform = from_origin(ulx, uly, pixel_size, pixel_size)

    profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': 'EPSG:6933',
        'transform': transform,
        'nodata': np.nan
    }
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(sms.astype(np.float32), 1)


def convert2tiff():
    os.makedirs(TIFF_DIR_PATH, exist_ok=True)
    file_names = [f for f in os.listdir(INPUT_DIR_PATH) if f.endswith(HDF5_SUFFIX)]
    for file_name in tqdm(file_names):
        file_path = os.path.join(INPUT_DIR_PATH, file_name)
        sms, date = extract(file_path)
        # 过滤非目标日期
        if not is_tgt_date(date):
            continue
        dst_path = os.path.join(TIFF_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        write2tiff(sms, dst_path)


def resample():
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    file_paths = glob.glob(os.path.join(TIFF_DIR_PATH, f"*{TIFF_SUFFIX}"))
    for file_path in tqdm(file_paths):
        dst_path = os.path.join(OUTPUT_DIR_PATH, os.path.basename(file_path))
        resample_tiff(file_path, STANDARD_GRID_PATH, dst_path)


def main():
    convert2tiff()
    resample()


if __name__ == "__main__":
    main()
