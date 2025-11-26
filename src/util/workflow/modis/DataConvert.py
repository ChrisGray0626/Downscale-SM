#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/25
"""
import glob
import os
from typing import List

import numpy as np
from affine import Affine
from osgeo import gdal
from pyproj import CRS

from Constant import NDVI_NAME, HDF4_SUFFIX, TIFF_SUFFIX
from util.DateUtil import is_valid_date, extract_date_from_modis_filename, handle_valid_date
from util.TiffUtil import write_tiff_from_transform
from util.workflow.common.Write import TiffWriter
from util.workflow.core.ContextKey import DATA_KEY, TRANSFORM_KEY, \
    GAP_VALUE_KEY, SCALE_FACTOR_KEY, DATA_NAME_KEY, DST_FILE_PATH_KEY, SRC_FILE_PATH_KEY, CRS_KEY, EPSG_CODE_KEY
from util.workflow.core.Base import BaseTask, Context, BaseFilter, BatchJob

__all__ = [
    "BatchConvert2TiffJob",
    "ValidDateFilter",
    "Reader",
    "DataProcessor",
    "TiffWriter",
    "ValidDateHandler",
]


class BatchConvert2TiffJob(BatchJob):
    def __init__(self,
                 src_dir_path_key: str,
                 dst_dir_path_key: str,
                 data_name_key: str = DATA_NAME_KEY,
                 gap_value_key: str = GAP_VALUE_KEY,
                 scale_factor_key: str = SCALE_FACTOR_KEY):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(
            ValidDateFilter(
                data_name_key=data_name_key,
                src_file_path_key=SRC_FILE_PATH_KEY
            ),
            Reader(
                src_file_path_key=SRC_FILE_PATH_KEY,
                data_key=DATA_KEY,
                transform_key=TRANSFORM_KEY,
                crs_key=CRS_KEY,
            ),
            DataProcessor(
                data_key=DATA_KEY,
                gap_value_key=gap_value_key,
                scale_factor_key=scale_factor_key
            ),
            TiffWriter(
                dst_file_path_key=DST_FILE_PATH_KEY,
                data_key=DATA_KEY,
                transform_key=TRANSFORM_KEY,
                epsg_code_key=EPSG_CODE_KEY,
                crs_key=CRS_KEY,
            ))

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{HDF4_SUFFIX}"))
        for src_file_path in src_file_paths:
            filename = os.path.basename(src_file_path)
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)

            dst_file_path = os.path.join(dst_dir_path, filename.replace(HDF4_SUFFIX, TIFF_SUFFIX))
            batch_context.set(DST_FILE_PATH_KEY, dst_file_path)

            batch_contexts.append(batch_context)

        return batch_contexts


class ValidDateFilter(BaseFilter):
    """
    Valid file path filter for MODIS data processing by date.

    Filtering logic:
    - NDVI data: No filtering applied, all data are preserved.
      Reason: NDVI has the maximum temporal resolution, and all other data
      need to be aligned to NDVI's time series. Therefore, all NDVI data
      must be retained to ensure temporal alignment.
    - Other data types (e.g., LST, Albedo): Filtered based on valid date list,
      only data with valid dates are preserved.
    """

    def __init__(self,
                 src_file_path_key: str = SRC_FILE_PATH_KEY,
                 data_name_key: str = DATA_NAME_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.data_name_key = data_name_key

    def filter(self, context: Context) -> bool:
        if context.get(self.data_name_key) == NDVI_NAME:
            return False
        src_file_path = context.get(self.src_file_path_key)
        filename = os.path.basename(src_file_path)
        date = extract_date_from_modis_filename(filename)

        return not is_valid_date(date)


class Reader(BaseTask):
    def __init__(self,
                 src_file_path_key: str = SRC_FILE_PATH_KEY,
                 data_key: str = DATA_KEY,
                 transform_key: str = TRANSFORM_KEY,
                 crs_key: str = CRS_KEY,
                 ):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.data_key = data_key
        self.transform_key = transform_key
        self.crs_key = crs_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        hdf_dataset = gdal.Open(src_path)
        sub_datasets = hdf_dataset.GetSubDatasets()

        sds_path = sub_datasets[0][0]
        ds = gdal.Open(sds_path)

        data = ds.ReadAsArray().astype(np.float32)
        transform = ds.GetGeoTransform()
        # Convert to Affine
        transform = Affine.from_gdal(*transform)
        projection = ds.GetProjection()
        crs = CRS.from_wkt(projection)

        context.set(self.data_key, data)
        context.set(self.transform_key, transform)
        context.set(self.crs_key, crs)

        return context


class DataProcessor(BaseTask):
    def __init__(self,
                 data_key: str = DATA_KEY,
                 gap_value_key: str = GAP_VALUE_KEY,
                 scale_factor_key: str = SCALE_FACTOR_KEY):
        super().__init__()
        self.data_key = data_key
        self.gap_value_key = gap_value_key
        self.scale_factor_key = scale_factor_key

    def execute(self, context) -> Context:
        data = context.get(self.data_key)
        gap_value = context.get(self.gap_value_key)
        scale_factor = context.get(self.scale_factor_key)
        data[data == gap_value] = np.nan
        data = data * scale_factor

        context.set(self.data_key, data)

        return context


class ValidDateHandler(BaseTask):

    def execute(self, context: Context) -> Context:
        handle_valid_date()

        return context
