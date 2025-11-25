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
from rasterio.transform import from_origin

from Constant import *
from util.DateUtil import is_valid_date
from util.TiffUtil import write_tiff_from_transform
from util.workflow.core.Base import BaseTask, Context, BaseFilter, Job, BatchJob
from util.workflow.common.Resample import BatchResampleTiffJob
from util.workflow.WorkflowConstant import SRC_FILE_PATH_KEY, DATA_KEY, GAP_VALUE_KEY, RAW_DIR_PATH_KEY, \
    DST_DIR_PATH_KEY, DATE_KEY, CONVERTED_DIR_PATH_KEY, RESAMPLED_DIR_PATH_KEY, REF_GRID_PATH_KEY

# Gap Value
GAP_VALUE = -9999

DATA_NAME = SM_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, CONVERTED_DIR_NAME)
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, RESOLUTION_36KM)


class BatchConvert2TiffJob(BatchJob):

    def __init__(self, src_dir_path_key: str, dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add([
            ValidDateFilter(
                src_file_path_key=SRC_FILE_PATH_KEY),
            Reader(
                src_file_path_key=SRC_FILE_PATH_KEY),
            DataProcessor(),
            Writer(
                dst_dir_path_key=dst_dir_path_key)
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


class Writer(BaseTask):

    def __init__(self, dst_dir_path_key: str = DST_DIR_PATH_KEY):
        super().__init__()
        self.dst_dir_path_key = dst_dir_path_key

    def execute(self, context: Context) -> Context:
        data = context.get(DATA_KEY)
        date = context.get(DATE_KEY)
        dst_dir_path = context.get(self.dst_dir_path_key)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")

        # EPSG:6933 EASE-Grid 2.0 Global 36km
        epsg_code = 6933
        pixel_size = 36032.22
        west = -17367530.44
        north = 7314540.83
        transform = from_origin(west, north, pixel_size, pixel_size)

        write_tiff_from_transform(
            data=data,
            transform=transform,
            dst_path=dst_file_path,
            epsg_code=epsg_code,
            nodata=np.nan,
            dtype=np.float32,
        )

        context.clear_local()

        return context


def main():
    job = Job()
    context = Context()
    job.add([
        BatchConvert2TiffJob(
            src_dir_path_key=RAW_DIR_PATH_KEY,
            dst_dir_path_key=CONVERTED_DIR_PATH_KEY
        ),
        BatchResampleTiffJob(
            src_dir_path_key=CONVERTED_DIR_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY,
            ref_grid_path_key=REF_GRID_PATH_KEY
        )
    ])
    # Convert to TIFF Config
    # Read Config
    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    # Data Processing Config
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    # Write Config
    context.set_global(CONVERTED_DIR_PATH_KEY, CONVERTED_DIR_PATH)

    # Resample Config
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)
    context.set_global(REF_GRID_PATH_KEY, REF_GRID_36KM_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
