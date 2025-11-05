#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle LST Data
  @Author Chris
  @Date 2025/5/5
"""
import glob
from collections import defaultdict

import numpy as np
from osgeo import gdal
from tqdm import tqdm

from constant import *
from util.tiff_util import merge_tiff, MODISDataProcessor, resample_tiff
from util.util import extract_date_from_modis_filename, is_tgt_date

gdal.UseExceptions()

# 单位：千米
RESOLUTION = 1
GAP_VALUE = 0
SCALE_FACTOR = 0.02

DIR_NAME = LST_NAME
INPUT_DIR_PATH = os.path.join(DATA_PATH, DIR_NAME)
TIFF_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, TIFF_DIR_NAME)
MERGED_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, MERGED_DIR_NAME)
OUTPUT_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, f"{RESOLUTION}km")
STANDARD_GRID_PATH = os.path.join(RESULT_PATH, f"Standard_Grid_{RESOLUTION}km{TIFF_SUFFIX}")


class LSTProcessor(MODISDataProcessor):

    def __init__(self, input_path, output_path):
        super().__init__(input_path, output_path)

    def process_data(self, data):
        data[data == GAP_VALUE] = np.nan
        data = data * SCALE_FACTOR
        return data


def convert2tiff():
    os.makedirs(TIFF_DIR_PATH, exist_ok=True)
    file_paths = glob.glob(os.path.join(INPUT_DIR_PATH, f"*{HDF4_SUFFIX}"))
    for file_path in tqdm(file_paths):
        filename = os.path.basename(file_path)
        date = extract_date_from_modis_filename(filename)
        if not is_tgt_date(date):
            continue
        output_file_path = os.path.join(TIFF_DIR_PATH, os.path.basename(file_path).replace(HDF4_SUFFIX, TIFF_SUFFIX))
        processor = LSTProcessor(file_path, output_file_path)
        processor.run()


def merge():
    os.makedirs(MERGED_DIR_PATH, exist_ok=True)

    # 按日期分类
    file_paths_group_by_date = defaultdict(list)
    for file_path in glob.glob(os.path.join(TIFF_DIR_PATH, f"*{TIFF_SUFFIX}")):
        filename = os.path.basename(file_path)
        date = extract_date_from_modis_filename(filename)
        file_paths_group_by_date[date].append(file_path)

    # 拼接并导出 TIFF
    for date, file_paths in tqdm(file_paths_group_by_date.items()):
        output_tiff = os.path.join(MERGED_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        merge_tiff(src_file_paths=file_paths, dst_path=output_tiff)


def resample():
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    file_paths = glob.glob(os.path.join(MERGED_DIR_PATH, f"*{TIFF_SUFFIX}"))
    for file_path in tqdm(file_paths):
        dst_path = os.path.join(OUTPUT_DIR_PATH, os.path.basename(file_path))
        resample_tiff(file_path, STANDARD_GRID_PATH, dst_path)


def main():
    convert2tiff()
    merge()
    resample()


if __name__ == "__main__":
    main()
