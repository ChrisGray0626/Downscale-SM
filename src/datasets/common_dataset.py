#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Common train/inference datasets with full variable set (grid, pos, features, SM).
  get_all: for pixel-level methods, returns all elements; inherit and extract what the method needs.
  __getitem__: left to subclasses (e.g. RF) to define.
  @Author Chris
  @Date 2026/1/30
"""
import threading

import numpy as np
from torch.utils.data import Dataset

from constants import *
from datasets.dataset import ModelDataStore, GridInfoStore
from utils.date_util import get_valid_dates

FEATURE_NAMES = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]


class CommonTrainDataset(Dataset):
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
        self.pos_flat = grid_info["pos"].reshape(-1, 2).astype(np.float64)

        xs_list, ys_list = [], []
        for date in dates:
            xs_date = np.stack(
                [self.data_store.get(name, date) for name in FEATURE_NAMES],
                axis=-1,
            )
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
        self.x_mean = np.nanmean(self.xs, axis=(0, 1, 2)).astype(np.float32)
        self.x_std = np.nanstd(self.xs, axis=(0, 1, 2)).astype(np.float32)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std

        self.y_mean = np.nanmean(self.ys)
        self.y_std = np.nanstd(self.ys)
        self.y_std = 1.0 if self.y_std == 0 else self.y_std
        self.ys = (self.ys - self.y_mean) / self.y_std

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return ys * self.y_std + self.y_mean

    def get_all(self):
        rows_grid = self.grid_info["rows"]
        cols_grid = self.grid_info["cols"]
        X_list, y_list, pos_list, date_list, rows_list, cols_list = [], [], [], [], [], []
        for i in range(len(self.dates)):
            valid = self.valid_masks[i]
            flat_valid = valid.reshape(-1)
            x_flat = self.xs[i].reshape(-1, len(FEATURE_NAMES))[flat_valid]
            y_flat = self.ys[i].reshape(-1)[flat_valid]
            pos_flat_i = self.pos_flat[flat_valid]
            rows_flat = rows_grid.reshape(-1)[flat_valid]
            cols_flat = cols_grid.reshape(-1)[flat_valid]
            X_list.append(x_flat.astype(np.float64))
            y_list.append(y_flat)
            pos_list.append(pos_flat_i)
            date_list.append(np.full(flat_valid.sum(), self.dates[i], dtype=object))
            rows_list.append(rows_flat)
            cols_list.append(cols_flat)
        return {
            DATE_NAME: np.concatenate(date_list, axis=0),
            POS_NAME: np.concatenate(pos_list, axis=0),
            ROW_NAME: np.concatenate(rows_list, axis=0),
            COL_NAME: np.concatenate(cols_list, axis=0),
            X_NAME: np.concatenate(X_list, axis=0),
            Y_NAME: np.concatenate(y_list, axis=0),
        }

    def __len__(self):
        return len(self.dates)


class CommonInferenceDataset(Dataset):

    def __init__(self, date: str, resolution: str):
        self.date = date
        self.resolution = resolution
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.train_dataset = CommonTrainDataset()

        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.grid_info = grid_info
        self.pos_flat = grid_info["pos"].reshape(-1, 2).astype(np.float64)
        self.rows_full = grid_info["rows"].flatten()
        self.cols_full = grid_info["cols"].flatten()

        xs = np.stack(
            [self.data_store.get(name, self.date) for name in FEATURE_NAMES],
            axis=-1,
        )
        self.xs = xs.reshape(self.H * self.W, -1).astype(np.float32)

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1)
        self.xs = self.xs[valid]
        self.rows = self.rows_full[valid]
        self.cols = self.cols_full[valid]
        self.pos = self.pos_flat[valid]

    def _norm(self):
        self.xs = (self.xs - self.train_dataset.x_mean) / self.train_dataset.x_std

    def get_all(self):
        return {
            DATE_NAME: self.date,
            POS_NAME: self.pos,
            ROW_NAME: self.rows,
            COL_NAME: self.cols,
            X_NAME: self.xs,
        }

    def denorm_y(self, ys: np.ndarray) -> np.ndarray:
        return self.train_dataset.denorm_y(ys)

    def __len__(self):
        return len(self.xs)
