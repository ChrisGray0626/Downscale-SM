#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description GeoTIFF Utility
  @Author Chris
  @Date 2025/5/8
"""
import glob
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from osgeo import gdal
from pyproj import CRS, Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds, reproject, Resampling
from scipy.interpolate import griddata, interp1d

from constants import *

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


def write_tiff(data,
               dst_file_path: str,
               transform: rasterio.Affine,
               epsg_code: int = None,
               crs: CRS = None,
               nodata: float = np.nan,
               dtype=None,
               ):
    data = np.asarray(data)

    if data.ndim != 2 and data.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D data_processor, got shape {data.shape}")

    # Ensure data_processor is 3-D for consistent processing
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


class TimeSeriesDataCache:
    """缓存时间序列数据，避免重复读取文件"""
    def __init__(self, src_dir_path: str):
        self.src_dir_path = src_dir_path
        self._cache = None
        self._file_date_map = None
        self._sorted_dates = None
        self._date_objects = None
        self._date_numeric = None
        self._profile = None
        self._height = None
        self._width = None

    def _load_data(self):
        """延迟加载时间序列数据"""
        if self._cache is not None:
            return

        # 获取所有源文件并按日期排序
        src_files = glob.glob(os.path.join(self.src_dir_path, f"*{TIFF_SUFFIX}"))
        if not src_files:
            raise ValueError(f"No TIFF files found in {self.src_dir_path}")

        # 提取日期并排序
        self._file_date_map = {}
        for file_path in src_files:
            filename = os.path.basename(file_path)
            date_str = filename.replace(TIFF_SUFFIX, '')
            if len(date_str) == 8:  # YYYYMMDD格式
                self._file_date_map[date_str] = file_path

        self._sorted_dates = sorted(self._file_date_map.keys())
        if not self._sorted_dates:
            raise ValueError("No valid date files found")

        # 读取第一个文件获取空间信息
        with rasterio.open(self._file_date_map[self._sorted_dates[0]]) as src:
            self._profile = src.profile.copy()
            self._height, self._width = src.height, src.width

        # 将日期字符串转换为datetime对象用于插值
        self._date_objects = [datetime.strptime(d, '%Y%m%d') for d in self._sorted_dates]
        self._date_numeric = np.array([(d - self._date_objects[0]).days for d in self._date_objects])

        # 读取所有数据到内存（按日期顺序）
        print(f"Loading {len(self._sorted_dates)} TIFF files for time series interpolation...")
        self._cache = np.full((len(self._sorted_dates), self._height, self._width), np.nan, dtype=np.float32)

        for i, date_str in enumerate(self._sorted_dates):
            file_path = self._file_date_map[date_str]
            with rasterio.open(file_path) as src:
                data = src.read(1)
                self._cache[i, :, :] = data

    def get_data_stack(self):
        """获取数据栈"""
        self._load_data()
        return self._cache

    def get_metadata(self):
        """获取元数据"""
        self._load_data()
        return {
            'profile': self._profile,
            'height': self._height,
            'width': self._width,
            'sorted_dates': self._sorted_dates,
            'date_objects': self._date_objects,
            'date_numeric': self._date_numeric
        }


def interpolate_single_date_tiff(src_dir_path: str,
                                target_date_str: str,
                                dst_file_path: str,
                                method: str = 'linear',
                                max_gap_days: int = 8):
    """
    对单个目标日期进行时间序列插值

    每次调用时动态查找目标日期前后 max_gap_days 范围内的文件，不使用缓存

    Parameters
    ----------
    src_dir_path : str
        源TIFF文件目录（包含所有1天分辨率的数据）
    target_date_str : str
        目标日期（YYYYMMDD格式）
    dst_file_path : str
        输出文件路径
    method : str, default 'linear'
        插值方法：'linear', 'cubic', 'nearest', 'time'
    max_gap_days : int, default 8
        最大允许的插值间隔（天数），前后各 max_gap_days 天
    """
    # 解析目标日期
    target_date = datetime.strptime(target_date_str, '%Y%m%d')

    # 计算需要查找的日期范围（目标日期前后各 max_gap_days 天）
    date_start = target_date - timedelta(days=max_gap_days)
    date_end = target_date + timedelta(days=max_gap_days)

    # 查找范围内的所有文件
    file_date_map = {}
    date_range = []
    current_date = date_start
    while current_date <= date_end:
        date_str = current_date.strftime('%Y%m%d')
        file_path = os.path.join(src_dir_path, f"{date_str}{TIFF_SUFFIX}")
        if os.path.exists(file_path):
            file_date_map[date_str] = file_path
            date_range.append(date_str)
        current_date += timedelta(days=1)

    if not file_date_map:
        raise ValueError(f"No data_processor files found in range [{date_start.strftime('%Y%m%d')}, {date_end.strftime('%Y%m%d')}] for target date {target_date_str}")

    # 检查目标日期本身是否有文件
    target_file_path = os.path.join(src_dir_path, f"{target_date_str}{TIFF_SUFFIX}")
    has_target_file = os.path.exists(target_file_path)

    # 读取第一个文件获取空间信息
    first_file = list(file_date_map.values())[0]
    with rasterio.open(first_file) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

    # 将日期字符串转换为datetime对象用于插值
    sorted_dates = sorted(file_date_map.keys())
    date_objects = [datetime.strptime(d, '%Y%m%d') for d in sorted_dates]
    date_numeric = np.array([(d - date_objects[0]).days for d in date_objects])
    target_numeric = (target_date - date_objects[0]).days

    # 读取范围内的所有数据到内存
    data_stack = np.full((len(sorted_dates), height, width), np.nan, dtype=np.float32)
    for i, date_str in enumerate(sorted_dates):
        file_path = file_date_map[date_str]
        with rasterio.open(file_path) as src:
            data = src.read(1)
            data_stack[i, :, :] = data

    # 初始化输出数组
    output_data = np.full((height, width), np.nan, dtype=np.float32)

    # 对每个像素进行时间序列插值
    for row in range(height):
        for col in range(width):
            pixel_series = data_stack[:, row, col]

            # 检查是否有有效数据
            valid_mask = ~np.isnan(pixel_series)
            if not np.any(valid_mask):
                continue  # 如果完全没有有效数据，保持NaN

            valid_indices = np.where(valid_mask)[0]
            valid_values = pixel_series[valid_indices]
            valid_dates = date_numeric[valid_indices]

            # 检查目标日期附近是否有数据
            if target_numeric in valid_dates:
                # 如果目标日期本身有数据，直接使用
                output_data[row, col] = pixel_series[np.where(date_numeric == target_numeric)[0][0]]
                continue

            # 检查最大间隔
            if len(valid_dates) < 2:
                continue  # 至少需要2个点才能插值

            # 找到目标日期前后的有效数据点
            before_idx = np.where(valid_dates < target_numeric)[0]
            after_idx = np.where(valid_dates > target_numeric)[0]

            if len(before_idx) == 0 or len(after_idx) == 0:
                continue  # 目标日期在数据范围外

            before_date = valid_dates[before_idx[-1]]
            after_date = valid_dates[after_idx[0]]
            gap_before = target_numeric - before_date
            gap_after = after_date - target_numeric

            # 检查是否超过最大间隔
            if gap_before > max_gap_days or gap_after > max_gap_days:
                continue

            # 执行插值
            if method == 'linear':
                # 线性插值
                interp_func = interp1d(valid_dates, valid_values,
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=np.nan)
                interpolated = interp_func(target_numeric)
                if not np.isnan(interpolated):
                    output_data[row, col] = interpolated
            elif method == 'cubic':
                # 三次样条插值（需要至少4个点）
                if len(valid_dates) >= 4:
                    interp_func = interp1d(valid_dates, valid_values,
                                           kind='cubic',
                                           bounds_error=False,
                                           fill_value=np.nan)
                    interpolated = interp_func(target_numeric)
                    if not np.isnan(interpolated):
                        output_data[row, col] = interpolated
                else:
                    # 回退到线性插值
                    interp_func = interp1d(valid_dates, valid_values,
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=np.nan)
                    interpolated = interp_func(target_numeric)
                    if not np.isnan(interpolated):
                        output_data[row, col] = interpolated
            elif method == 'time':
                # 时间加权插值（距离越近权重越大）
                total_weight = 0
                weighted_sum = 0
                for v_date, v_value in zip(valid_dates, valid_values):
                    distance = abs(v_date - target_numeric)
                    if distance == 0:
                        weighted_sum = v_value
                        total_weight = 1
                        break
                    weight = 1.0 / (distance + 1)  # 加1避免除零
                    weighted_sum += weight * v_value
                    total_weight += weight
                if total_weight > 0:
                    output_data[row, col] = weighted_sum / total_weight
            elif method == 'nearest':
                # 最近邻插值
                nearest_idx = np.argmin(np.abs(valid_dates - target_numeric))
                output_data[row, col] = valid_values[nearest_idx]

    # 写入输出文件
    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
    profile.update(dtype='float32', count=1, nodata=np.nan)
    with rasterio.open(dst_file_path, 'w', **profile) as dst:
        dst.write(output_data, 1)


def interpolate_time_series_tiff(src_dir_path: str,
                                 dst_dir_path: str,
                                 target_dates: List[str],
                                 method: str = 'linear',
                                 max_gap_days: int = 32):
    """
    对多个目标日期进行时间序列插值（兼容旧接口）

    注意：此函数内部使用单日期插值函数，每次单独检索文件
    """
    os.makedirs(dst_dir_path, exist_ok=True)

    # 对每个目标日期进行插值
    print(f"Interpolating {len(target_dates)} target dates...")
    for target_date_str in target_dates:
        try:
            dst_file_path = os.path.join(dst_dir_path, f"{target_date_str}{TIFF_SUFFIX}")
            interpolate_single_date_tiff(
                src_dir_path=src_dir_path,
                target_date_str=target_date_str,
                dst_file_path=dst_file_path,
                method=method,
                max_gap_days=max_gap_days
            )
            print(f"  Interpolated and saved: {target_date_str}")
        except ValueError as e:
            print(f"  Warning: {e}, skipping {target_date_str}")
            continue

    print("Time series interpolation completed!")
