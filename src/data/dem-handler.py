#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle DEM Data
  @Author Chris
  @Date 2025/5/6
"""
import os

from tqdm import tqdm

from constant import DATA_PATH, RESULT_PATH, TIFF_SUFFIX, DEM_NAME, MERGED_DIR_NAME
from util.tiff_util import merge_tiff, resample_tiff
from util.util import unzip_file

DIR_NAME = DEM_NAME
DIR_PATH = os.path.join(DATA_PATH, DIR_NAME)
UNZIP_DIR_PATH = os.path.join(DIR_PATH, "unzip")
MERGED_FILE_PATH = os.path.join(DIR_PATH, MERGED_DIR_NAME, f"merged{TIFF_SUFFIX}")

# 单位：千米
RESOLUTION = 1
STANDARD_GRID_PATH = os.path.join(RESULT_PATH, f"Standard_Grid_{RESOLUTION}km{TIFF_SUFFIX}")
OUTPUT_PATH = os.path.join(RESULT_PATH, DEM_NAME, f"{RESOLUTION}km{TIFF_SUFFIX}")


def unzip():
    os.makedirs(UNZIP_DIR_PATH, exist_ok=True)
    for file_name in tqdm(os.listdir(DIR_PATH)):
        if not file_name.endswith(".zip"):
            continue
        file_path = os.path.join(DIR_PATH, file_name)
        unzip_file(file_path, UNZIP_DIR_PATH)


def merge():
    os.makedirs(os.path.dirname(os.path.join(DIR_PATH, MERGED_DIR_NAME)), exist_ok=True)
    merge_tiff(MERGED_FILE_PATH, src_dir_path=UNZIP_DIR_PATH)


def resample():
    os.makedirs(os.path.join(RESULT_PATH, DIR_NAME), exist_ok=True)
    resample_tiff(MERGED_FILE_PATH, STANDARD_GRID_PATH, OUTPUT_PATH)


def main():
    unzip()
    merge()
    resample()


if __name__ == "__main__":
    main()
