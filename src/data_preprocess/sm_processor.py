#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Solid Moisture Data
  @Author Chris
  @Date 2025/4/25
"""
import glob
from datetime import datetime, timedelta
from typing import List

import h5py as h5
import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_origin

from constants import *
from datasets.base_data_store import BaseDataStore
from utils.raster_util import write_tiff, read_tiff_meta, read_tiff_data
from utils.workflow.common.Resampler import BatchResampleTiffJob
from utils.workflow.common.Writer import TiffWriter
from utils.workflow.core.base import BaseTask, Context, Job, BatchJob
from utils.workflow.core.context_key import SRC_FILE_PATH_KEY, DATA_KEY, GAP_VALUE_KEY, RAW_DIR_PATH_KEY, \
    DST_DIR_PATH_KEY, DATE_KEY, CONVERTED_DIR_PATH_KEY, RESAMPLED_DIR_PATH_KEY, REF_GRID_PATH_KEY, TRANSFORM_KEY, \
    DST_FILE_PATH_KEY, CRS_KEY, INTERPOLATED_DIR_PATH_KEY, WINDOW_SIZE_KEY

# Gap Value
GAP_VALUE = -9999

# Time Series Interpolation Window Size
WINDOW_SIZE = 2

DATA_NAME = SM_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, CONVERTED_DIR_NAME)
INTERPOLATED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, INTERPOLATED_DIR_NAME)
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

    # Time Series Interpolation Config
    context.set_global(WINDOW_SIZE_KEY, WINDOW_SIZE)
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


class TimeSeriesInterpolationJob(BatchJob):

    def __init__(self,
                 src_dir_path_key: str = CONVERTED_DIR_PATH_KEY,
                 dst_dir_path_key: str = INTERPOLATED_DIR_PATH_KEY,
                 ):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add([
            TimeSeriesInterpolator(
                src_dir_path_key=src_dir_path_key,
                dst_file_path_key=DST_FILE_PATH_KEY,
            )
        ])

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)
        window_size = context.get(WINDOW_SIZE_KEY)

        dates = [os.path.splitext(f)[0] for f in os.listdir(src_dir_path) if f.endswith(TIFF_SUFFIX)]

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []

        for date in dates:
            batch_context = context.global_copy()
            batch_context.set(self.src_dir_path_key, src_dir_path)
            batch_context.set(DATE_KEY, date)
            batch_context.set(WINDOW_SIZE_KEY, window_size)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}"))
            batch_contexts.append(batch_context)

        return batch_contexts


class TimeSeriesInterpolator(BaseTask):

    def __init__(self,
                 src_dir_path_key: str = SRC_FILE_PATH_KEY,
                 dst_file_path_key: str = DST_FILE_PATH_KEY,
                 date_key: str = DATE_KEY,
                 window_size_key: str = WINDOW_SIZE_KEY):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_file_path_key = dst_file_path_key
        self.date_key = date_key
        self.window_size_key = window_size_key

    def execute(self, context: Context) -> Context:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_file_path = context.get(self.dst_file_path_key)
        date = context.get(self.date_key)
        window_size = context.get(self.window_size_key)

        tgt_file_path = os.path.join(src_dir_path, f"{date}{TIFF_SUFFIX}")

        self._time_series_interpolate(
            src_dir_path=src_dir_path,
            tgt_file_path=tgt_file_path,
            dst_file_path=dst_file_path,
            window_size=window_size
        )

        return context

    @staticmethod
    def _time_series_interpolate(src_dir_path: str,
                                 tgt_file_path: str,
                                 dst_file_path: str,
                                 window_size: int):
        tgt_date_str = os.path.splitext(os.path.basename(tgt_file_path))[0]
        tgt_date = datetime.strptime(tgt_date_str, '%Y%m%d')
        data_store = ConvertedStore()

        date_start = tgt_date - timedelta(days=window_size)
        date_end = tgt_date + timedelta(days=window_size)

        all_dates = sorted([os.path.splitext(f)[0] for f in os.listdir(src_dir_path)
                            if f.endswith(TIFF_SUFFIX)])

        window_dates = [(date_start + timedelta(days=i)).strftime('%Y%m%d')
                        for i in range((date_end - date_start).days + 1)]

        dates = [d for d in window_dates if d in all_dates]

        if tgt_date_str not in dates:
            raise ValueError(f"Target date {tgt_date_str} not found in available dates")

        date_objects = [datetime.strptime(d, '%Y%m%d') for d in dates]
        time_points = np.array([(d - date_objects[0]).days for d in date_objects])
        tgt_idx = dates.index(tgt_date_str)
        tgt_time = time_points[tgt_idx]

        data_stack = np.stack([data_store.get(d, cache_used=True) for d in dates])

        transform, crs, height, width = read_tiff_meta(tgt_file_path)

        window_mask = np.abs(time_points - tgt_time) <= window_size
        window_times = time_points[window_mask]
        window_values = data_stack[window_mask, :, :]

        # Linear Regression Interpolation
        output_data = np.full((height, width), np.nan, dtype=np.float32)
        for row in range(height):
            for col in range(width):
                pixel_window = window_values[:, row, col]
                valid_mask = ~np.isnan(pixel_window)

                if not np.any(valid_mask):
                    continue

                valid_values = pixel_window[valid_mask]
                valid_times = window_times[valid_mask]

                if len(valid_times) < 2:
                    continue

                coeff = np.polyfit(valid_times, valid_values, 1)
                interpolated = np.polyval(coeff, tgt_time)
                if not np.isnan(interpolated):
                    output_data[row, col] = interpolated

        write_tiff(output_data, dst_file_path, transform=transform, crs=crs, nodata=np.nan)


class ConvertedStore(BaseDataStore[np.ndarray]):

    def __init__(self):
        super().__init__()
        self.src_dir_path = CONVERTED_DIR_PATH

    def get(self, date_str: str, cache_used: bool = True) -> np.ndarray:
        return self._get(date_str, lambda: self._load(date_str), cache_used=cache_used)

    def _load(self, date_str: str) -> np.ndarray:
        file_path = os.path.join(self.src_dir_path, f"{date_str}{TIFF_SUFFIX}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return read_tiff_data(file_path).astype(np.float32)


if __name__ == "__main__":
    main()
