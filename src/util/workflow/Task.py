#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/19
"""
import glob
import os
from collections import defaultdict

import numpy as np
from osgeo import gdal
from tqdm import tqdm

from constant import TIFF_SUFFIX
from util.tiff_util import merge_tiff, resample_tiff
from util.util import extract_date_from_modis_filename

from util.workflow.Base import *


class HDF4Reader(BaseReader):

    def execute(self, context) -> Context:
        input_path = context.get(INPUT_PATH_KEY)
        hdf_dataset = gdal.Open(input_path)
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
        output_path = context.get(OUTPUT_PATH_KEY)
        data = context.get(DATA_KEY)
        transform = context.get(TRANSFORM_KEY)
        projection = context.get(PROJECTION_KEY)
        x_size = context.get(X_SIZE_KEY)
        y_size = context.get(Y_SIZE_KEY)

        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_path,
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


class Merger(BaseTask):

    def execute(self, context) -> Context:
        tiff_dir_path = context.get(TIFF_DIR_PATH_KEY)
        merged_dir_path = context.get(MERGED_DIR_PATH_KEY)

        # 按日期分类
        file_paths_group_by_date = defaultdict(list)
        for file_path in glob.glob(os.path.join(tiff_dir_path, f"*{TIFF_SUFFIX}")):
            filename = os.path.basename(file_path)
            date = extract_date_from_modis_filename(filename)
            file_paths_group_by_date[date].append(file_path)

        if not merged_dir_path:
            os.makedirs(merged_dir_path, exist_ok=True)

        for date, file_paths in tqdm(file_paths_group_by_date.items()):
            output_tiff = os.path.join(merged_dir_path, f"{date}{TIFF_SUFFIX}")
            merge_tiff(dst_path=output_tiff, src_file_paths=file_paths)

        return context


class Resampler(BaseTask):

    def execute(self, context) -> Context:
        merged_dir_path = context.get(MERGED_DIR_PATH_KEY)
        output_dir_path = context.get(OUTPUT_DIR_PATH_KEY)
        standard_grid_path = context.get(STANDARD_GRID_PATH_KEY)

        if not merged_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)

        file_paths = glob.glob(os.path.join(merged_dir_path, f"*{TIFF_SUFFIX}"))
        for file_path in tqdm(file_paths):
            dst_path = os.path.join(output_dir_path, os.path.basename(file_path))
            resample_tiff(file_path, standard_grid_path, dst_path)

        return context
