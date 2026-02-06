#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Common train/inference datasets with full variable set (grid, lons, lats, features, SM).
  get_all: for pixel-level methods, returns all elements; inherit and extract what the method needs.
  __getitem__: left to subclasses (e.g. RF) to define.
  @Author Chris
  @Date 2026/1/30
"""
import threading

import numpy as np
from torch.utils.data import Dataset

from constants import *
from datasets.common_data_store import ModelDataStore, GridInfoStore
from utils.date_util import get_valid_dates

__all__ = [
    "BaseTrainDataset",
    "BaseInferenceDataset",
]

FEATURE_NAMES = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]


class BaseTrainDataset(Dataset):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, flat: bool = True, filter_valid: bool = True):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, flat: bool = True, filter_valid: bool = True):
        if hasattr(self, "_initialized"):
            return
        self.flat = flat
        self.filter_valid = filter_valid

        self.resolution = RESOLUTION_36KM
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._load_data()
        self._build_valid_masks()
        self._norm()

        if self.flat:
            self._stage_flat()
        if self.filter_valid:
            self._stage_filter_valid()

        self._initialized = True

    def _load_data(self):
        dates = get_valid_dates()
        self.dates = np.array(dates, dtype=object)

        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.grid_info = grid_info
        self.pos = np.asarray(grid_info["pos"], dtype=np.float32)
        self.rows = grid_info["rows"]
        self.cols = grid_info["cols"]

        xs_list, ys_list = [], []
        for date in dates:
            xs_date = np.stack([self.data_store.get(name, date) for name in FEATURE_NAMES], axis=-1, )
            ys_date = self.data_store.get(SM_NAME, date)
            xs_list.append(xs_date)
            ys_list.append(ys_date)

        self.xs = np.array(xs_list, dtype=np.float32)
        self.ys = np.array(ys_list, dtype=np.float32)

    def _build_valid_masks(self):
        valid_masks = []
        for i in range(len(self.dates)):
            valid_xs = ~np.isnan(self.xs[i]).any(axis=-1)
            valid_ys = ~np.isnan(self.ys[i])
            valid_masks.append(valid_xs & valid_ys)
        self.valid_masks = np.array(valid_masks, dtype=bool)

    def _norm(self):
        self.x_mean = np.nanmean(self.xs, axis=(0, 1, 2)).astype(np.float32)
        self.x_std = np.nanstd(self.xs, axis=(0, 1, 2)).astype(np.float32)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std

        self.y_mean = np.nanmean(self.ys)
        self.y_std = np.nanstd(self.ys)
        self.y_std = 1.0 if self.y_std == 0 else self.y_std
        self.ys = (self.ys - self.y_mean) / self.y_std

    def _stage_flat(self):
        dates = self.dates
        pos_1d = self.pos.reshape(-1, 2)
        rows_1d = self.rows.reshape(-1)
        cols_1d = self.cols.reshape(-1)
        x_list, y_list, pos_list, date_list, rows_list, cols_list = [], [], [], [], [], []
        for i in range(len(dates)):
            n = self.H * self.W
            x_list.append(self.xs[i].reshape(n, -1).astype(np.float32))
            y_list.append(self.ys[i].reshape(-1))
            pos_list.append(pos_1d)
            date_list.append(np.full(n, dates[i], dtype=object))
            rows_list.append(rows_1d)
            cols_list.append(cols_1d)
        self.xs = np.concatenate(x_list, axis=0)
        self.ys = np.concatenate(y_list, axis=0)
        self.dates = np.concatenate(date_list, axis=0)
        self.pos = np.concatenate(pos_list, axis=0)
        self.rows = np.concatenate(rows_list, axis=0)
        self.cols = np.concatenate(cols_list, axis=0)

    def _stage_filter_valid(self):
        valid_flat = np.concatenate([self.valid_masks[i].reshape(-1) for i in range(len(self.valid_masks))])
        if self.xs.ndim == 2:
            self.xs = self.xs[valid_flat]
            self.ys = self.ys[valid_flat]
            self.dates = self.dates[valid_flat]
            self.pos = self.pos[valid_flat]
            self.rows = self.rows[valid_flat]
            self.cols = self.cols[valid_flat]
        else:
            n, T = self.H * self.W, len(self.dates)
            self.xs = self.xs.reshape(-1, self.xs.shape[-1])[valid_flat]
            self.ys = self.ys.reshape(-1)[valid_flat]
            self.dates = np.repeat(self.dates, n)[valid_flat]
            self.pos = np.tile(self.pos.reshape(-1, 2), (T, 1))[valid_flat]
            self.rows = np.tile(self.rows.reshape(-1), T)[valid_flat]
            self.cols = np.tile(self.cols.reshape(-1), T)[valid_flat]

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_std + self.y_mean

    def get_all(self):
        return {
            DATE_NAME: self.dates,
            LONGITUDE_NAME: self.pos[:, 0],
            LATITUDE_NAME: self.pos[:, 1],
            ROW_NAME: self.rows,
            COL_NAME: self.cols,
            X_NAME: self.xs,
            Y_NAME: self.ys,
        }


class BaseInferenceDataset(Dataset):

    def __init__(self, date: str, resolution: str, flat: bool = True, filter_valid: bool = True):
        self.date = date
        self.resolution = resolution
        self.flat = flat
        self.filter_valid = filter_valid
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.train_dataset = BaseTrainDataset(flat=flat, filter_valid=filter_valid)
        self._load_data()
        self._build_valid_mask()
        self._norm()
        if self.flat:
            self._stage_flat()
        if self.filter_valid:
            self._stage_filter_valid()

    def _load_data(self):
        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.grid_info = grid_info
        self.pos = np.asarray(grid_info["pos"], dtype=np.float32)
        self.rows = grid_info["rows"]
        self.cols = grid_info["cols"]
        xs = np.stack(
            [self.data_store.get(name, self.date) for name in FEATURE_NAMES],
            axis=-1,
        )
        self.xs = xs.astype(np.float32)

    def _build_valid_mask(self):
        self.valid = ~np.isnan(self.xs).any(axis=-1)

    def _norm(self):
        x_mean = self.train_dataset.x_mean
        x_std = self.train_dataset.x_std
        self.xs = (self.xs - x_mean) / x_std

    def _stage_flat(self):
        n = self.H * self.W
        self.xs = self.xs.reshape(n, -1)
        self.pos = self.pos.reshape(-1, 2)
        self.rows = self.rows.reshape(-1)
        self.cols = self.cols.reshape(-1)

    def _stage_filter_valid(self):
        valid = self.valid.reshape(-1)
        if self.xs.ndim == 2:
            self.xs = self.xs[valid]
            self.pos = self.pos[valid]
            self.rows = self.rows[valid]
            self.cols = self.cols[valid]
        else:
            self.xs = self.xs.reshape(self.H * self.W, -1)[valid]
            self.pos = self.pos.reshape(-1, 2)[valid]
            self.rows = self.rows.reshape(-1)[valid]
            self.cols = self.cols.reshape(-1)[valid]

    def get_all(self):
        return {
            DATE_NAME: self.date,
            LONGITUDE_NAME: self.pos[:, 0],
            LATITUDE_NAME: self.pos[:, 1],
            ROW_NAME: self.rows,
            COL_NAME: self.cols,
            X_NAME: self.xs,
        }

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return self.train_dataset.denorm_y(ys)

    def __len__(self):
        return self.H * self.W
