#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Random Forest Dataset - Date-based (not pixel-based)
@Author Chris
@Date 2025/12/12
"""
import threading

import numpy as np
from torch.utils.data import Dataset

from constants import *
from datasets.dataset import ModelDataStore, GridInfoStore
from utils.data_store import TiffStore
from utils.date_util import get_valid_dates


class RFTrainDataset(Dataset):
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
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.grid_info = grid_info

        xs_list, ys_list = [], []

        for date in dates:
            xs_date = np.stack([
                self.data_store.get(name, date) for name in
                [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
            ], axis=-1)
            ys_date = self.data_store.get(SM_NAME, date)

            xs_list.append(xs_date)
            ys_list.append(ys_date)

        self.xs = np.array(xs_list, dtype=np.float32)
        self.ys = np.array(ys_list, dtype=np.float32)

    def _filter_valid(self):
        valid_masks = []
        for i in range(len(self.dates)):
            xs_date = self.xs[i]
            ys_date = self.ys[i]
            valid_xs = ~np.isnan(xs_date).any(axis=-1)
            valid_ys = ~np.isnan(ys_date)
            valid_mask = valid_xs & valid_ys
            valid_masks.append(valid_mask)
        self.valid_masks = np.array(valid_masks, dtype=bool)

    def _norm(self):
        self.x_mean = np.nanmean(self.xs, axis=(0, 1, 2)).astype(np.float32)  # (5,)
        self.x_std = np.nanstd(self.xs, axis=(0, 1, 2)).astype(np.float32)    # (5,)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std

        self.y_mean = np.nanmean(self.ys)
        self.y_std = np.nanstd(self.ys)
        self.y_std = 1.0 if self.y_std == 0 else self.y_std
        self.ys = (self.ys - self.y_mean) / self.y_std

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_std + self.y_mean

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = str(self.dates[idx])
        xs = self.xs[idx]
        ys = self.ys[idx]
        valid_mask = self.valid_masks[idx]
        return date, xs, ys, valid_mask


class RFInferenceDataset(Dataset):

    def __init__(self, date: str, resolution: str):
        self.date = date
        self.resolution = resolution
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.train_dataset = RFTrainDataset()

        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.grid_info = grid_info

        # Features: (H, W, 5)
        xs = np.stack([
            self.data_store.get(name, self.date) for name in
            [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
        ], axis=-1)

        self.xs = xs.reshape(self.H * self.W, -1).astype(np.float32)
        self.rows_full = grid_info["rows"].flatten()
        self.cols_full = grid_info["cols"].flatten()

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1)

        self.xs = self.xs[valid]
        self.rows = self.rows_full[valid]
        self.cols = self.cols_full[valid]

    def _norm(self):
        self.xs = (self.xs - self.train_dataset.x_mean) / self.train_dataset.x_std

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return self.train_dataset.denorm_y(ys)

    def __len__(self):
        return len(self.xs)


class RFResultStore(TiffStore):
    def __init__(self, resolution: str):
        base_dir = RF_DIR_PATH
        super().__init__(base_dir, resolution)
