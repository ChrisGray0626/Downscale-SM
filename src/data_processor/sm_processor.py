#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Solid Moisture Data
  @Author Chris
  @Date 2025/4/25
"""
import glob
from typing import List

import h5py as h5
import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_origin

from constants import *
from utils.date_util import is_valid_date, get_valid_dates
from utils.tiff_util import interpolate_single_date_tiff
from utils.workflow.common.Resampler import BatchResampleTiffJob
from utils.workflow.common.Writer import TiffWriter
from utils.workflow.core.base import BaseTask, Context, BaseFilter, Job, BatchJob
from utils.workflow.core.context_key import SRC_FILE_PATH_KEY, DATA_KEY, GAP_VALUE_KEY, RAW_DIR_PATH_KEY, \
    DST_DIR_PATH_KEY, DATE_KEY, CONVERTED_DIR_PATH_KEY, RESAMPLED_DIR_PATH_KEY, REF_GRID_PATH_KEY, TRANSFORM_KEY, \
    DST_FILE_PATH_KEY, CRS_KEY, INTERPOLATED_DIR_PATH_KEY

# Gap Value
GAP_VALUE = -9999

DATA_NAME = SM_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, CONVERTED_DIR_NAME)
INTERPOLATED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, "Interpolated")
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, RESOLUTION_36KM)


def main():
    job = Job()
    context = Context()
    job.add([
        # BatchConvert2TiffJob(
        #     src_dir_path_key=RAW_DIR_PATH_KEY,
        #     dst_dir_path_key=CONVERTED_DIR_PATH_KEY
        # ),
        TimeSeriesInterpolationJob(
            src_dir_path_key=CONVERTED_DIR_PATH_KEY,
            dst_dir_path_key=INTERPOLATED_DIR_PATH_KEY,
            method='linear',
            max_gap_days=2
        ),
        BatchResampleTiffJob(
            src_dir_path_key=INTERPOLATED_DIR_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY,
        )
    ])
    # Convert to TIFF Config
    # Read Config
    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    # Data Processing Config
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    # Write Config
    context.set_global(CONVERTED_DIR_PATH_KEY, CONVERTED_DIR_PATH)
    context.set_global(INTERPOLATED_DIR_PATH_KEY, INTERPOLATED_DIR_PATH)
    context.set_global(TRANSFORM_KEY, build_6933_transform())
    context.set_global(CRS_KEY, CRS.from_epsg(6933))

    # Resample Config
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)
    context.set_global(REF_GRID_PATH_KEY, REF_GRID_36KM_PATH)

    job.run(context)


class BatchConvert2TiffJob(BatchJob):

    def __init__(self, src_dir_path_key: str, dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add([
            Reader(),
            DataProcessor(),
            DstFilePathBuilder(),
            TiffWriter(),
        ])

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{HDF5_SUFFIX}"))
        dst_dir_path = context.get(self.dst_dir_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []

        for src_file_path in src_file_paths:
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)
            batch_context.set(DST_DIR_PATH_KEY, dst_dir_path)
            batch_context.set(GAP_VALUE_KEY, GAP_VALUE)
            batch_contexts.append(batch_context)

        return batch_contexts


class ValidDateFilter(BaseFilter):

    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key

    def filter(self, context: Context) -> bool:
        src_file_path = context.get(self.src_file_path_key)
        date = os.path.basename(src_file_path)[13:21]

        return not is_valid_date(date)


class Reader(BaseTask):

    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key

    def execute(self, context: Context) -> Context:
        src_file_path = context.get(self.src_file_path_key)
        f = h5.File(src_file_path, "r")
        data = f["Soil_Moisture_Retrieval_Data_AM/soil_moisture"][:]
        date = os.path.basename(src_file_path)[13:21]
        context.set(DATA_KEY, data)
        context.set(DATE_KEY, date)

        return context


class DataProcessor(BaseTask):

    def __init__(self, data_key: str = DATA_KEY, gap_value_key: str = GAP_VALUE_KEY):
        super().__init__()
        self.data_key = data_key
        self.gap_value_key = gap_value_key

    def execute(self, context: Context) -> Context:
        data = context.get(self.data_key)
        gap_value = context.get(self.gap_value_key)
        data[data == gap_value] = np.nan
        context.set(self.data_key, data)

        return context


class DstFilePathBuilder(BaseTask):

    def __init__(self,
                 dst_dir_path_key: str = DST_DIR_PATH_KEY,
                 dst_file_path_key: str = DST_FILE_PATH_KEY,
                 date_key: str = DATE_KEY,
                 ):
        super().__init__()
        self.dst_dir_path_key = dst_dir_path_key
        self.dst_file_path_key = dst_file_path_key
        self.date_key = date_key

    def execute(self, context: Context) -> Context:
        date = context.get(self.date_key)
        dst_dir_path = context.get(self.dst_dir_path_key)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        context.set(self.dst_file_path_key, dst_file_path)

        return context


def build_6933_transform() -> Affine:
    # EPSG:6933 EASE-Grid 2.0 Global 36km
    pixel_size = 36032.22
    west = -17367530.44
    north = 7314540.83
    transform = from_origin(west, north, pixel_size, pixel_size)

    return transform


# TODO TimeSeriesInterpolationJob
class TimeSeriesInterpolationJob(BatchJob):

    def __init__(self,
                 src_dir_path_key: str = CONVERTED_DIR_PATH_KEY,
                 dst_dir_path_key: str = INTERPOLATED_DIR_PATH_KEY,
                 method: str = 'linear',
                 max_gap_days: int = 4):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.method = method
        self.max_gap_days = max_gap_days
        self.add([
            TimeSeriesInterpolator(
                src_dir_path_key=src_dir_path_key,
                dst_file_path_key=DST_FILE_PATH_KEY,
                date_key=DATE_KEY,
                method_key='interpolation_method',
                max_gap_days_key='max_gap_days'
            )
        ])

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        target_dates = get_valid_dates()

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []

        for target_date_str in target_dates:
            batch_context = context.global_copy()
            batch_context.set(self.src_dir_path_key, src_dir_path)
            batch_context.set(DATE_KEY, target_date_str)
            batch_context.set('interpolation_method', self.method)
            batch_context.set('max_gap_days', self.max_gap_days)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, f"{target_date_str}{TIFF_SUFFIX}"))
            batch_contexts.append(batch_context)

        return batch_contexts


class TimeSeriesInterpolator(BaseTask):

    def __init__(self,
                 src_dir_path_key: str = CONVERTED_DIR_PATH_KEY,
                 dst_file_path_key: str = DST_FILE_PATH_KEY,
                 date_key: str = DATE_KEY,
                 method_key: str = 'interpolation_method',
                 max_gap_days_key: str = 'max_gap_days'):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_file_path_key = dst_file_path_key
        self.date_key = date_key
        self.method_key = method_key
        self.max_gap_days_key = max_gap_days_key

    def execute(self, context: Context) -> Context:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_file_path = context.get(self.dst_file_path_key)
        target_date_str = context.get(self.date_key)
        method = context.get(self.method_key, 'linear')
        max_gap_days = context.get(self.max_gap_days_key)

        interpolate_single_date_tiff(
            src_dir_path=src_dir_path,
            target_date_str=target_date_str,
            dst_file_path=dst_file_path,
            method=method,
            max_gap_days=max_gap_days
        )

        return context


if __name__ == "__main__":
    main()
