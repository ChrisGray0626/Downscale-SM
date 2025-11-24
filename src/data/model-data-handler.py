#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Model Data (Labeled Data & Prediction Data)
  @Author Chris
  @Date 2025/6/11
"""

import pandas as pd
from tqdm import tqdm

from Constant import *
from util.tiff_util import read_tiff, read_tiff_data
from util.util import convert_projection, get_tgt_dates

TRAIN_MODE = "train"
PREDICTION_MODE = "prediction"

MODE = TRAIN_MODE

if MODE == TRAIN_MODE:
    RESOLUTION = RESOLUTION_1KM
    STANDARD_GRID_PATH = REF_GRID_1KM_PATH
    OUTPUT_DIR_PATH = LABELED_DATA_DIR_PATH
elif MODE == PREDICTION_MODE:
    RESOLUTION = RESOLUTION_1KM
    STANDARD_GRID_PATH = REF_GRID_1KM_PATH
    OUTPUT_DIR_PATH = PRED_DATA_DIR_PATH
else:
    raise ValueError("Mode must be determined")

DEM_PATH = os.path.join(RESULT_PATH, DEM_NAME, f"{RESOLUTION}{TIFF_SUFFIX}")
NDVI_DIR_PATH = os.path.join(RESULT_PATH, NDVI_NAME, RESOLUTION)


def handle_lons_lats():
    data, lons, lats = read_tiff(STANDARD_GRID_PATH)
    lons = lons.ravel()
    lats = lats.ravel()

    return lons, lats


def handle_dem():
    dem = read_tiff_data(DEM_PATH)
    dem = dem.ravel()

    return dem


def main():
    lons, lats = handle_lons_lats()
    xs, ys = convert_projection(lons, lats, src_espg_code=4326, dst_epsg_code=6933)
    dem = handle_dem()
    dates = tqdm(get_tgt_dates())
    for date in dates:
        df = pd.DataFrame({
            DATE_NAME: date,
            LONGITUDE_NAME: lons,
            LATITUDE_NAME: lats,
            PROJ_X_NAME: xs,
            PROJ_Y_NAME: ys,
            DEM_NAME: dem,
        })
        data_names = [
            NDVI_NAME, LST_NAME, SM_NAME, ALBEDO_NAME, PRECIPITATION_NAME
        ]
        for data_name in data_names:
            file_path = os.path.join(RESULT_PATH, data_name, RESOLUTION, f"{date}{TIFF_SUFFIX}")
            data = read_tiff_data(file_path)
            df[data_name] = data.ravel()
        # # 清洗数据
        # if MODE == TRAIN_MODE:
        #     # 清洗所有自变量与因变量中的缺失值
        #     df = df.dropna()
        # elif MODE == PREDICTION_MODE:
        #     # 只清洗自变量中的缺失值
        #     df = df.dropna(subset=[NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME])
        # 清洗空值
        if df.empty:
            print(f"\n No data for date: {date} \n")
            continue
        os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR_PATH, f"{date}.csv")
        df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
