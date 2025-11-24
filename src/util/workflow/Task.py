#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Task
  @Author Chris
  @Date 2025/11/19
"""

import os
import numpy as np
from osgeo import gdal
from typing import Optional

from Constant import ZIP_SUFFIX, NDVI_NAME
from util.DateUtil import handle_valid_date, extract_date_from_modis_filename, is_valid_date
from util.TiffUtil import merge_tiff, resample_tiff
from util.util import unzip_file
from util.workflow.Base import *
from util.workflow.WorkflowConstant import *

__all__ = [
    'Decompressor',
    'HDF4Reader',
    'TiffWriter',
    'MODISDataProcessor',
    'TiffMerger',
    'TiffResampler',
    'ValidDateHandler',
    'ValidDateFilter',
]


class Decompressor(BaseTask):
    def __init__(self, src_file_path_key: str, dst_dir_path_key: str, name: Optional[str] = None):
        super().__init__(name)
        self.src_file_path_key = src_file_path_key
        self.dst_dir_path_key = dst_dir_path_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        if src_path.endswith(ZIP_SUFFIX):
            unzip_file(src_path, dst_dir_path)

        return context


class HDF4Reader(BaseTask):
    def __init__(self, src_file_path_key: str, name: Optional[str] = None):
        super().__init__(name)
        self.src_file_path_key = src_file_path_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        hdf_dataset = gdal.Open(src_path)
        sub_datasets = hdf_dataset.GetSubDatasets()

        sds_path = sub_datasets[0][0]
        ds = gdal.Open(sds_path)

        data = ds.ReadAsArray().astype(np.float32)
        transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        x_size = ds.RasterXSize
        y_size = ds.RasterYSize

        context.set(DATA_KEY, data)
        context.set(TRANSFORM_KEY, transform)
        context.set(PROJECTION_KEY, projection)
        context.set(X_SIZE_KEY, x_size)
        context.set(Y_SIZE_KEY, y_size)

        return context


class TiffWriter(BaseTask):
    def __init__(self,
                 dst_file_path_key: str,
                 data_key: str,
                 transform_key: str,
                 projection_key: str,
                 x_size_key: str,
                 y_size_key: str,
                 name: Optional[str] = None):
        super().__init__(name)
        self.dst_file_path_key = dst_file_path_key
        self.data_key = data_key
        self.transform_key = transform_key
        self.projection_key = projection_key
        self.x_size_key = x_size_key
        self.y_size_key = y_size_key

    def execute(self, context):
        dst_path = context.get(self.dst_file_path_key)
        data = context.get(self.data_key)
        transform = context.get(self.transform_key)
        projection = context.get(self.projection_key)
        x_size = context.get(self.x_size_key)
        y_size = context.get(self.y_size_key)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            dst_path,
            x_size,
            y_size,
            1,
            gdal.GDT_Float32
        )

        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(np.nan)

        out_dataset.SetGeoTransform(transform)
        out_dataset.SetProjection(projection)

        out_dataset.FlushCache()

        return context


class MODISDataProcessor(BaseTask):
    def __init__(self,
                 data_key: str,
                 gap_value_key: str,
                 scale_factor_key: str,
                 name: Optional[str] = None):
        super().__init__(name)
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


class TiffMerger(BaseTask):
    def __init__(self,
                 dst_file_path_key: str,
                 src_file_paths_key: Optional[str] = None,
                 src_dir_path_key: Optional[str] = None,
                 name: Optional[str] = None):
        super().__init__(name)
        if src_file_paths_key is None and src_dir_path_key is None:
            raise ValueError("Either src_file_paths_key or src_dir_path_key must be provided.")
        self.src_file_paths_key = src_file_paths_key
        self.src_dir_path_key = src_dir_path_key
        self.dst_file_path_key = dst_file_path_key

    def execute(self, context) -> Context:
        dst_file_path = context.get(self.dst_file_path_key)
        src_file_paths = context.get(self.src_file_paths_key, None)
        src_dir_path = context.get(self.src_dir_path_key, None)

        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

        merge_tiff(
            dst_file_path=dst_file_path,
            src_file_paths=src_file_paths,
            src_dir_path=src_dir_path
        )

        return context


class TiffResampler(BaseTask):
    def __init__(self,
                 src_file_path_key: str,
                 ref_grid_path_key: str,
                 dst_file_path_key: str,
                 name: Optional[str] = None):
        super().__init__(name)
        self.src_file_path_key = src_file_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_file_path_key = dst_file_path_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        ref_grid_path = context.get(self.ref_grid_path_key)
        dst_path = context.get(self.dst_file_path_key)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        resample_tiff(src_path, ref_grid_path, dst_path)

        return context


class ValidDateHandler(BaseTask):

    def execute(self, context: Context) -> Context:
        handle_valid_date()

        return context


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

    def __init__(self, data_name: str, src_file_path_key: str, name: Optional[str] = None):
        super().__init__(name)
        self.data_name = data_name
        self.src_file_path_key = src_file_path_key

    def filter(self, context: Context) -> bool:
        if self.data_name == NDVI_NAME:
            return False
        src_file_path = context.get(self.src_file_path_key)
        filename = os.path.basename(src_file_path)
        date = extract_date_from_modis_filename(filename)

        return not is_valid_date(date)
