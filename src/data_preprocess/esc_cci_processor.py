#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle ESA CCI Soil Moisture Data
  @Author Chris
  @Date 2026/1/27
"""
import glob
from typing import List

import numpy as np
from pyproj import CRS
import xarray as xr

from constants import *
from utils.util import build_transform_from_lonlat
from utils.workflow.common.Resampler import BatchMultiResampleTiffJob, ResolutionConfig
from utils.workflow.common.Writer import TiffWriter
from utils.workflow.core.base import Job, Context, BatchJob, BaseTask
from utils.workflow.core.context_key import *

GAP_VALUE = -9999

DATA_NAME = ESA_CCI_NAME
ROOT_DIR_PATH = os.path.join(DATA_DIR_PATH, DATA_NAME)
RAW_DIR_PATH = os.path.join(ROOT_DIR_PATH, RAW_NAME)
CONVERTED_DIR_PATH = os.path.join(ROOT_DIR_PATH, CONVERTED_DIR_NAME)
RESAMPLED_DIR_PATH = os.path.join(ROOT_DIR_PATH)


def main():
    job = Job()
    context = Context()

    job.add([
        BatchConvertNetCDF2TiffJob(
            src_dir_path_key=RAW_DIR_PATH_KEY,
            dst_dir_path_key=CONVERTED_DIR_PATH_KEY,
        ),
        BatchMultiResampleTiffJob(
            src_dir_path_key=CONVERTED_DIR_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY,
        ),
    ])

    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    context.set_global(DATA_NAME_KEY, DATA_NAME)
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    context.set_global(CONVERTED_DIR_PATH_KEY, CONVERTED_DIR_PATH)
    context.set_global(RESOLUTION_CONFIGS_KEY, [
        ResolutionConfig(
            resolution_km=25,
            ref_grid_path=REF_GRID_25KM_PATH,
        ),
        ResolutionConfig(
            resolution_km=36,
            ref_grid_path=REF_GRID_36KM_PATH,
        ),
    ])
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)

    job.run(context)


class BatchConvertNetCDF2TiffJob(BatchJob):
    def __init__(self,
                 src_dir_path_key: str,
                 dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(
            Reader(),
            DataProcessor(),
            DstFilePathBuilder(),
            TiffWriter()
        )

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{NETCDF_SUFFIX}"))

        for src_file_path in src_file_paths:
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)
            batch_context.set(DST_DIR_PATH_KEY, dst_dir_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class Reader(BaseTask):

    def __init__(self,
                 src_file_path_key: str = SRC_FILE_PATH_KEY,
                 data_key: str = DATA_KEY,
                 date_key: str = DATE_KEY,
                 transform_key: str = TRANSFORM_KEY,
                 crs_key: str = CRS_KEY,
                 ):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.data_key = data_key
        self.date_key = date_key
        self.transform_key = transform_key
        self.crs_key = crs_key

    def execute(self, context) -> Context:
        src_file_path = context.get(self.src_file_path_key)
        ds = xr.open_dataset(src_file_path)

        data = ds['sm'].isel(time=0).values.astype(np.float32)

        lons = ds['lon'].values
        lats = ds['lat'].values

        # Flip data if latitudes are in ascending order
        if len(lats) > 1 and lats[0] < lats[-1]:
            data = np.flipud(data)
            lats = np.flip(lats)

        time = ds['time'].values[0]
        date = str(time)[:10].replace('-', '')

        lons = np.asarray(lons, dtype=np.float32)
        lats = np.asarray(lats, dtype=np.float32)
        transform = build_transform_from_lonlat(lons, lats)

        ds.close()

        context.set(self.data_key, data)
        context.set(self.date_key, date)
        context.set(self.transform_key, transform)
        context.set(self.crs_key, CRS.from_epsg(4326))

        return context


class DataProcessor(BaseTask):
    def __init__(self,
                 data_key: str = DATA_KEY,
                 gap_value_key: str = GAP_VALUE_KEY,
                 ):
        super().__init__()
        self.data_key = data_key
        self.gap_value_key = gap_value_key

    def execute(self, context) -> Context:
        data = context.get(self.data_key)
        gap_value = context.get(self.gap_value_key)

        data[data == gap_value] = np.nan

        context.set(DATA_KEY, data)
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
        dst_dir_path = context.get(self.dst_dir_path_key)
        date = context.get(self.date_key)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        context.set(self.dst_file_path_key, dst_file_path)

        return context


if __name__ == "__main__":
    main()
