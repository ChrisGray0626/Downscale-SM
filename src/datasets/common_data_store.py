#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Dataset for Task
  @Author Chris
  @Date 2025/11/12
"""
import threading
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import *
from datasets.base_data_store import BaseDataStore, BaseTiffStore
from utils.date_util import get_valid_dates
from utils.raster_util import read_tiff, read_tiff_data, read_tiff_meta

__all__ = [
    'TrainDataset',
    'ModelDataStore',
    'GridInfoStore',
    'InsituStatsStore',
    'InsituStore',
]


class TrainDataset(Dataset):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.resolution = RESOLUTION_36KM
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._load_data()
        self._filter_valid()
        self._norm()

        self._initialized = True

    def _load_data(self):
        dates = get_valid_dates()
        self.dates = np.array(dates, dtype=object)

        grid_info = self.grid_info_store.get()

        xs_list, ys_list, insitu_list = [], [], []

        for date in dates:
            xs_date = np.stack([
                self.data_store.get(name, date) for name in
                [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
            ], axis=-1)
            ys_date = self.data_store.get(SM_NAME, date)[:, :, np.newaxis]
            insitu_date = self.data_store.get(IN_SITU_NAME, date)

            xs_list.append(xs_date)
            ys_list.append(ys_date)
            insitu_list.append(insitu_date)

        xs = np.array(xs_list, dtype=np.float32)
        ys = np.array(ys_list, dtype=np.float32)
        insitu = np.array(insitu_list, dtype=np.float32)

        date_num = len(dates)
        H = grid_info["H"]
        W = grid_info["W"]
        self.xs = xs.reshape(date_num * H * W, -1).astype(np.float32)
        self.ys = ys.reshape(date_num * H * W, -1).astype(np.float32)
        self.insitus = insitu.reshape(date_num * H * W).astype(np.float32)

        pos_grid = grid_info["pos"]
        pos_expanded = np.tile(pos_grid[np.newaxis, :, :, :], (date_num, 1, 1, 1))
        self.pos = pos_expanded.reshape(date_num * H * W, -1).astype(np.float32)

        rows_grid = grid_info["rows"]
        cols_grid = grid_info["cols"]
        rows_expanded = np.tile(rows_grid[np.newaxis, :, :], (date_num, 1, 1))
        cols_expanded = np.tile(cols_grid[np.newaxis, :, :], (date_num, 1, 1))
        self.rows = rows_expanded.reshape(date_num * H * W).astype(np.int32)
        self.cols = cols_expanded.reshape(date_num * H * W).astype(np.int32)

        self.date_indices = np.repeat(np.arange(date_num), H * W).astype(np.int32)

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1) & ~np.isnan(self.ys).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.ys = self.ys[valid, 0]
        self.insitus = self.insitus[valid]
        self.rows = self.rows[valid]
        self.cols = self.cols[valid]
        self.date_indices = self.date_indices[valid]

        self.insitu_masks = (~np.isnan(self.insitus)).astype(np.float32)

    def _norm(self):
        self.x_mean = self.xs.mean(axis=0).astype(np.float32)
        self.x_std = self.xs.std(axis=0).astype(np.float32)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std

        self.y_mean = self.ys.mean()
        self.y_std = self.ys.std()
        self.ys = (self.ys - self.y_mean) / self.y_std

    def denorm_y(self, ys):
        if isinstance(ys, torch.Tensor):
            y_std = torch.tensor(self.y_std, dtype=ys.dtype, device=ys.device)
            y_mean = torch.tensor(self.y_mean, dtype=ys.dtype, device=ys.device)
            return ys * y_std + y_mean
        else:
            return ys * self.y_std + self.y_mean

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).float()
        ys = torch.tensor(self.ys[idx], dtype=torch.float32)
        pos = torch.from_numpy(self.pos[idx]).float()
        date_idx = self.date_indices[idx]
        date = str(self.dates[date_idx])

        return xs, ys, pos, date


class InsituStatsStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution
        self.insitu_store = InsituStore(resolution=self.resolution)

    def get(self, date: str, cache_used: bool = True) -> np.ndarray:
        return self._get(date, lambda: self._load(date), cache_used=cache_used)

    def _load(self, date: str) -> np.ndarray:
        insitus = self.insitu_store.get(date)

        return self._calc_insitu_stats_from_data(insitus)

    @staticmethod
    def _calc_insitu_stats_from_data(insitus: np.ndarray) -> np.ndarray:
        valid_insitu = insitus[~np.isnan(insitus)]
        if valid_insitu.size > 0:
            insitu_stats = np.array([
                np.mean(valid_insitu),
                np.std(valid_insitu),
                np.percentile(valid_insitu, 25),
                np.percentile(valid_insitu, 75),
            ], dtype=np.float32)
        else:
            insitu_stats = np.zeros(4, dtype=np.float32)
        return insitu_stats


class ModelDataStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution
        self._tiff_stores: Dict[str, BaseTiffStore] = {}

    def _tiff_store(self, name: str) -> BaseTiffStore:
        if name not in self._tiff_stores:
            base_dir = os.path.join(PROCESSED_DIR_PATH, name)
            self._tiff_stores[name] = BaseTiffStore(base_dir=base_dir, resolution=self.resolution)
        return self._tiff_stores[name]

    def get(self, name: str, date: Optional[str] = None, cache_used: bool = True) -> np.ndarray:
        if name == DEM_NAME:
            return DEMStore(self.resolution).get()
        return self._tiff_store(name).get(date, cache_used=cache_used)

    def _load_dem(self) -> np.ndarray:
        file_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, self.resolution, f"{DEM_NAME}{TIFF_SUFFIX}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return read_tiff_data(file_path).astype(np.float32)


class DEMStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution

    def get(self, cache_used: bool = True) -> np.ndarray:
        key = self.resolution
        return self._get(key, lambda: self._load(), cache_used=cache_used)

    def _load(self) -> np.ndarray:
        file_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, self.resolution, f"{DEM_NAME}{TIFF_SUFFIX}")
        return read_tiff_data(file_path).astype(np.float32)


class SMAPStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = os.path.join(PROCESSED_DIR_PATH, SM_NAME)
        super().__init__(base_dir, resolution)


class InsituStore(BaseTiffStore):
    def __init__(self, resolution: str):
        base_dir = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME)
        super().__init__(base_dir, resolution)


class GridInfoStore(BaseDataStore[Dict]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution

    def get(self, cache_used: bool = True) -> Dict:
        key = self.resolution
        return self._get(key, lambda: self._load(), cache_used=cache_used)

    def _load(self) -> Dict:
        if self.resolution == RESOLUTION_1KM:
            ref_path = REF_GRID_1KM_PATH
        else:
            ref_path = REF_GRID_36KM_PATH

        transform, crs, H, W = read_tiff_meta(ref_path)
        _, lons, lats = read_tiff(ref_path, dst_epsg_code=4326)
        rows, cols = np.meshgrid(np.arange(H, dtype=np.int32),
                                 np.arange(W, dtype=np.int32), indexing="ij")
        grid_info = {
            "lons": lons,
            "lats": lats,
            "rows": rows,
            "cols": cols,
            "H": H,
            "W": W,
            "pos": np.stack([lons, lats], axis=-1).astype(np.float32),
            "transform": transform,
            "crs": crs,
        }
        return grid_info
