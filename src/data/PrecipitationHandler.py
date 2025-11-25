#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Precipitation Data
  @Author Chris
  @Date 2025/6/5
"""
import glob
from typing import List

import numpy as np
import xarray as xr

from Constant import *
from util.DateUtil import is_valid_date
from util.TiffUtil import write_lonlat_tiff
from util.workflow.Base import BaseTask, Context, BaseFilter, BatchJob, Job
from util.workflow.Job import ResolutionConfig
from util.workflow.WorkflowConstant import SRC_FILE_PATH_KEY, DATA_KEY, LONGITUDE_KEY, LATITUDE_KEY, DATE_KEY, \
    DST_DIR_PATH_KEY, RAW_DIR_PATH_KEY, CONVERTED_DIR_PATH_KEY, RESOLUTION_CONFIGS_KEY, \
    REF_GRID_PATH_KEY, RESAMPLED_DIR_PATH_KEY

# 单位：千米
RESOLUTION = 1
DATA_NAME = PRECIPITATION_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, CONVERTED_DIR_NAME)
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME)


class BatchConvert2TiffJob(BatchJob):

    def __init__(self, src_dir_path_key: str, dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add([
            ValidDateFilter(
                src_file_path_key=SRC_FILE_PATH_KEY
            ),
            Reader(
                src_file_path_key=SRC_FILE_PATH_KEY
            ),
            Writer(
                dst_dir_path_key=dst_dir_path_key
            )
        ])

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{NETCDF_SUFFIX}"))
        dst_dir_path = context.get(self.dst_dir_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []

        for src_file_path in src_file_paths:
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)
            batch_context.set(DST_DIR_PATH_KEY, dst_dir_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class ValidDateFilter(BaseFilter):

    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key

    def filter(self, context: Context) -> bool:
        src_file_path = context.get(self.src_file_path_key)

        with xr.open_dataset(src_file_path) as ds:
            date = ds['time'].values[0]
        # 日期转换
        date = str(date)[:10].replace('-', '')

        return not is_valid_date(date)


class Reader(BaseTask):

    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key

    def execute(self, context: Context) -> Context:
        src_file_path = context.get(self.src_file_path_key)

        # 使用 context manager 确保数据集正确关闭
        with xr.open_dataset(src_file_path) as ds:
            data = ds['precipitation'].isel(time=0)
            lons = ds['lon'].values
            lats = ds['lat'].values
            date = ds['time'].values[0]

        # 翻转数据（使用 copy=False 如果可能，但这里需要翻转所以必须复制）
        data = data.transpose()
        data = data[::-1, :]
        data = np.asarray(data, dtype=np.float32)

        # 日期转换
        date = str(date)[:10].replace('-', '')

        context.set(DATA_KEY, data)
        context.set(LONGITUDE_KEY, lons)
        context.set(LATITUDE_KEY, lats)
        context.set(DATE_KEY, date)

        return context


class Writer(BaseTask):

    def __init__(self, dst_dir_path_key: str = DST_DIR_PATH_KEY):
        super().__init__()
        self.dst_dir_path_key = dst_dir_path_key

    def execute(self, context: Context) -> Context:
        data = context.get(DATA_KEY)
        lons = context.get(LONGITUDE_KEY)
        lats = context.get(LATITUDE_KEY)
        date = context.get(DATE_KEY)
        dst_dir_path = context.get(self.dst_dir_path_key)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")

        write_lonlat_tiff(
            data=data,
            lons=lons,
            lats=lats,
            dst_path=dst_file_path,
            epsg_code=4326,
            nodata=np.nan,
            dtype=np.float32
        )

        context.clear_local()

        return context


def main():
    job = Job()
    context = Context()

    from util.workflow.Job import BatchMultiResampleTiffJob
    job.add([
        BatchConvert2TiffJob(
            src_dir_path_key=RAW_DIR_PATH_KEY,
            dst_dir_path_key=CONVERTED_DIR_PATH_KEY,
        ),
        BatchMultiResampleTiffJob(
            resolution_configs_key=RESOLUTION_CONFIGS_KEY,
            src_dir_path_key=CONVERTED_DIR_PATH_KEY,
            ref_grid_path_key=REF_GRID_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY,
        )
    ])

    # Convert to TIFF Config
    # Read Config
    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    # Write Config
    context.set_global(CONVERTED_DIR_PATH_KEY, CONVERTED_DIR_PATH)

    # Multi-Resolution Resample Config
    # Multi Resample Config
    context.set_global(RESOLUTION_CONFIGS_KEY, [
        ResolutionConfig(
            resolution_km=1,
            ref_grid_path=REF_GRID_1KM_PATH,
        ),
        ResolutionConfig(
            resolution_km=36,
            ref_grid_path=REF_GRID_36KM_PATH,
        ),
    ])
    # Resample Config
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
