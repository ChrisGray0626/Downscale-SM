#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description GeoTIFF Utility
  @Author Chris
  @Date 2025/5/8
"""
import glob
import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from osgeo import gdal
from pyproj import CRS, Transformer
from rasterio.transform import rowcol, from_origin
from rasterio.warp import transform_bounds, reproject, Resampling
from scipy.interpolate import griddata

from Constant import LONGITUDE_NAME, LATITUDE_NAME, PROJ_X_NAME, PROJ_Y_NAME, ROW_NAME, COL_NAME, DATA_NAME

gdal.UseExceptions()


def merge_tiff(dst_file_path: str, src_dir_path: str = None, src_file_paths: list = None):
    if src_dir_path is None and src_file_paths is None:
        raise ValueError("Either src_dir_path or src_file_paths must be provided.")
    # 如果提供了 src_file_paths，则忽略 src_dir_path
    if src_dir_path is not None:
        if not os.path.exists(src_dir_path):
            raise ValueError(f"Directory {src_dir_path} does not exist.")
        if not os.path.isdir(src_dir_path):
            raise ValueError(f"{src_dir_path} is not a directory.")
        src_file_paths = glob.glob(os.path.join(src_dir_path, '*'))
    if not src_file_paths:
        raise ValueError("No files found in the specified directory.")
    gdal.Warp(dst_file_path, src_file_paths, format="GTiff")


def resample_tiff(src_path, ref_grid_path, dst_path):
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

    with rasterio.open(ref_grid_path) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_nodata = ref.nodata

    # 空数组用于接收插值后的新数据
    dst_data = np.empty(ref_shape, dtype=src_data.dtype)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata,
        # dst_nodata=ref_nodata,
    )

    with rasterio.open(
            dst_path,
            "w",
            driver="GTiff",
            height=ref.height,
            width=ref.width,
            count=1,
            dtype=dst_data.dtype,
            crs=ref_crs,
            transform=ref_transform
    ) as dst:
        dst.write(dst_data, 1)


def show_tiff(file_path: str, dst_epsg_code: int = 4326):
    with rasterio.open(file_path) as dataset:
        # 读取数据（第1波段）
        data = dataset.read(1)
        # 获取仿射变换（地理坐标到像素坐标转换）
        transform = dataset.transform
        # 获取坐标参考系统（CRS）
        crs = dataset.crs
        # 获取边界（范围）
        bounds = dataset.bounds
        # 获取像素大小
        res = dataset.res
    if crs is None:
        crs = CRS.from_epsg(dst_epsg_code)
    # 转换为 EPSG:4326
    if dst_epsg_code != crs.to_epsg():
        bounds = transform_bounds(crs, CRS.from_epsg(dst_epsg_code), *bounds)
    data = np.ma.masked_invalid(data)
    d = data[~np.isnan(data)]
    print("Transform: ", transform)
    print("Bounds: ", bounds)
    print("Resolution (pixel size): ", res)
    print("Data shape: ", data.shape)
    # 显示高程图像
    plt.imshow(data, cmap='terrain')
    plt.colorbar()
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


def read_tiff_data(file_path: str):
    with rasterio.open(file_path) as src:
        data = src.read(1)

    return data


def read_tiff(file_path: str, dst_epsg_code: int = 4326):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        transform_affine = src.transform
        src_crs = src.crs  # 源投影 CRS
        width = src.width
        height = src.height

    # 构建行列索引网格
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    # 使用仿射变换将行列号转换为原始投影下的坐标
    xs, ys = rasterio.transform.xy(transform_affine, rows, cols, offset='center')

    # Reshape 为二维数组
    xs = np.array(xs).reshape((height, width))
    ys = np.array(ys).reshape((height, width))

    # 投影转换（原投影 -> EPSG:4326）
    if dst_epsg_code != src_crs.to_epsg():
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(dst_epsg_code), always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    return data, xs, ys


def interpolate_tiff(data, lons, lats, grid_path, dst_path, espg_code: int = 4326):
    with rasterio.open(grid_path) as grid:
        grid_data = grid.read(1)
        grid_transform = grid.transform
        grid_crs = grid.crs
        grid_width = grid.width
        grid_height = grid.height
    transformer = Transformer.from_crs(CRS.from_epsg(espg_code), grid_crs, always_xy=True)

    df = pd.DataFrame(
        {
            DATA_NAME: data,
            LONGITUDE_NAME: lons,
            LATITUDE_NAME: lats,
        }
    )
    # 投影转换
    df[PROJ_X_NAME], df[PROJ_Y_NAME] = transformer.transform(df[LONGITUDE_NAME].values, df[LATITUDE_NAME].values)
    # 计算像元行列索引
    df[ROW_NAME], df[COL_NAME] = rowcol(grid_transform, df[PROJ_X_NAME], df[PROJ_Y_NAME])
    # 像元聚合
    df = df.groupby(by=[ROW_NAME, COL_NAME])[DATA_NAME].mean()
    # 插入数据
    for (row, col), value in df.items():
        if 0 <= row < grid_height and 0 <= col < grid_width:
            grid_data[row, col] = value

    with rasterio.open(
            dst_path,
            'w',
            driver="GTiff",
            width=grid_width,
            height=grid_height,
            count=1,
            dtype=grid_data.dtype,
            crs=grid_crs,
            transform=grid_transform,
            nodata=np.nan,
    ) as dst:
        dst.write(grid_data, 1)


def interpolate(src_data, src_lon, src_lat, grid_path, dst_path, src_espg_code=4326):
    with rasterio.open(grid_path) as grid:
        dst_crs = grid.crs
        dst_transform = grid.transform
        dst_width = grid.width
        dst_height = grid.height
        dst_profile = grid.profile
    src_crs = CRS.from_epsg(src_espg_code)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(src_lon, src_lat)
    # affine transform 是左上角为原点，计算中心坐标
    col, row = np.meshgrid(np.arange(dst_width), np.arange(dst_height))
    grid_x, grid_y = rasterio.transform.xy(dst_transform, row, col, offset='center')
    grid_x = np.array(grid_x).reshape(dst_height, dst_width)
    grid_y = np.array(grid_y).reshape(dst_height, dst_width)

    data_interp = griddata(
        (x.flatten(), y.flatten()),
        src_data.flatten(),
        (grid_x, grid_y),
        method='linear'
    )
    data_interp_nearest = griddata(
        (x.flatten(), y.flatten()),
        src_data.flatten(),
        (grid_x, grid_y),
        method='nearest'
    )
    # 填补缺失值
    data_interp = np.where(np.isnan(data_interp), data_interp_nearest, data_interp)

    dst_profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        dst.write(data_interp.astype(np.float32), 1)


def write_tiff_from_transform(data,
                              dst_file_path: str,
                              transform: rasterio.Affine,
                              epsg_code: int = None,
                              crs: CRS = None,
                              nodata: float = np.nan,
                              dtype=None,
                              ):
    data = np.asarray(data)

    if data.ndim != 2 and data.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D data, got shape {data.shape}")

    # Ensure data is 3-D for consistent processing
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    band, height, width = data.shape

    if epsg_code is not None:
        crs = CRS.from_epsg(epsg_code)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': band,
        'crs': crs,
        'transform': transform,
        'dtype': dtype or data.dtype,
        'nodata': nodata,
    }
    with rasterio.open(dst_file_path, 'w', **profile) as dst:
        dst.write(data)
