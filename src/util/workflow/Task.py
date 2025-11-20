#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/19
"""
import glob
import os

import numpy as np
from osgeo import gdal

from constant import TIFF_SUFFIX
from util.tiff_util import merge_tiff, resample_tiff
from util.workflow.Base import *


class HDF4Reader(BaseReader):

    def execute(self, context) -> Context:
        src_path = context.get(SRC_FILE_PATH_KEY)
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


class TiffWriter(BaseWriter):

    def execute(self, context):
        dst_path = context.get(DST_FILE_PATH_KEY)
        data = context.get(DATA_KEY)
        transform = context.get(TRANSFORM_KEY)
        projection = context.get(PROJECTION_KEY)
        x_size = context.get(X_SIZE_KEY)
        y_size = context.get(Y_SIZE_KEY)

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

    def execute(self, context) -> Context:
        data = context.get(DATA_KEY)
        gap_value = context.get(GAP_VALUE_KEY)
        scale_factor = context.get(SCALE_FACTOR_KEY)
        data[data == gap_value] = np.nan
        data = data * scale_factor

        context.set(DATA_KEY, data)

        return context


class TiffMerger(BaseTask):

    def execute(self, context) -> Context:
        src_file_paths = context.get(SRC_FILE_PATHS_KEY)
        dst_path = context.get(DST_FILE_PATH_KEY)

        merge_tiff(
            src_file_paths=src_file_paths,
            dst_path=dst_path,
        )

        return context


class TiffResampler(BaseTask):

    def execute(self, context) -> Context:
        src_path = context.get(SRC_FILE_PATH_KEY)
        ref_grid_path = context.get(REF_GRID_PATH_KEY)
        dst_path = context.get(DST_FILE_PATH_KEY)

        resample_tiff(src_path, ref_grid_path, dst_path)

        return context
