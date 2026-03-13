#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Pixel DDPM dataset with DDPM_Image-conditioned residual targets and heterogeneity features
  @Author Chris
  @Date 2026/3/9
"""

import threading
from typing import Tuple

import numpy as np
import torch
from rasterio.warp import reproject, Resampling

from constants import *
from datasets.base_data_store import BaseTiffStore
from datasets.base_dataset import BaseInferenceDataset, BaseTrainDataset
from datasets.common_data_store import GridInfoStore, ModelDataStore
from ddpm_image.image_dataset import DDPMImageInferenceResultStore

HETEROGENEITY_FEAT_NAMES = [NDVI_NAME, LST_NAME, DEM_NAME]
MIN_SUBCELL_VALID_RATIO = 0.5


class DDPMPixelTrainDataset(BaseTrainDataset):

    def __init__(self):
        super().__init__(flat=True, filter_valid=False)

        feature_valid_mask = ~np.isnan(self.xs)
        valid_y = ~np.isnan(self.ys)

        self.xs_valid_mask = feature_valid_mask[valid_y].astype(np.float32)
        self.xs = self.xs[valid_y]
        self.ys = self.ys[valid_y]
        self.dates = self.dates[valid_y]
        self.pos = self.pos[valid_y]
        self.rows = self.rows[valid_y]
        self.cols = self.cols[valid_y]

        self.image_baselines = self._load_norm_image_baselines()
        self.heterogeneity_features, heter_valid_mask = self._load_heterogeneity_features()
        self.residuals = self.ys - self.image_baselines

        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)
        self.image_baselines = np.nan_to_num(self.image_baselines, nan=0.0, posinf=0.0, neginf=0.0)
        self.heterogeneity_features = np.nan_to_num(self.heterogeneity_features, nan=0.0, posinf=0.0, neginf=0.0)
        self.residuals = np.nan_to_num(self.residuals, nan=0.0, posinf=0.0, neginf=0.0)
        self.xs = np.concatenate(
            [self.xs, self.image_baselines[:, np.newaxis], self.heterogeneity_features],
            axis=1,
        )

        baseline_valid_mask = np.ones((self.xs.shape[0], 1), dtype=np.float32)
        self.xs_valid_mask = np.concatenate(
            [self.xs_valid_mask, baseline_valid_mask, heter_valid_mask.astype(np.float32)],
            axis=1,
        )

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int) -> Tuple:
        date = str(self.dates[idx])
        pos = self.pos[idx].astype(np.float32)
        xs = self.xs[idx].astype(np.float32)
        residual = np.asarray(self.residuals[idx], dtype=np.float32)
        valid_mask = self.xs_valid_mask[idx].astype(np.float32)

        xs_t = torch.from_numpy(xs)
        residual_t = torch.tensor(residual, dtype=torch.float32)
        pos_t = torch.from_numpy(pos)
        valid_mask_t = torch.from_numpy(valid_mask)

        return date, pos_t, xs_t, residual_t, valid_mask_t

    def _load_norm_image_baselines(self) -> np.ndarray:
        store = DDPMImageInferenceResultStore(resolution=self.resolution)
        baselines = np.zeros_like(self.ys, dtype=np.float32)
        for date in np.unique(self.dates):
            baseline_map = store.get(str(date)).astype(np.float32)
            baseline_map = (baseline_map - self.y_mean) / self.y_std
            mask = self.dates == date
            baselines[mask] = baseline_map[self.rows[mask], self.cols[mask]]
        return baselines

    def _load_heterogeneity_features(self) -> Tuple[np.ndarray, np.ndarray]:
        store = HeterogeneityFeatureStore(resolution=self.resolution)
        feat_num = len(HETEROGENEITY_FEAT_NAMES)
        features = np.full((self.ys.shape[0], feat_num), np.nan, dtype=np.float32)
        for date in np.unique(self.dates):
            heter_map = store.get(str(date)).astype(np.float32)
            mask = self.dates == date
            features[mask] = heter_map[self.rows[mask], self.cols[mask], :]
        valid_mask = ~np.isnan(features)
        return features, valid_mask


class DDPMPixelInferenceDataset(BaseInferenceDataset):

    def __init__(self, date: str, resolution: str = RESOLUTION_36KM):
        self.date = date
        self.resolution = resolution
        super().__init__(date=self.date, resolution=self.resolution, flat=True, filter_valid=False)
        self.image_baseline = self._load_norm_image_baseline()
        self.heterogeneity_features = self._load_heterogeneity_features()
        self.image_baseline = np.nan_to_num(self.image_baseline, nan=0.0, posinf=0.0, neginf=0.0)
        heter_valid_mask = ~np.isnan(self.heterogeneity_features)
        self.heterogeneity_features = np.nan_to_num(self.heterogeneity_features, nan=0.0, posinf=0.0, neginf=0.0)
        self.xs = np.concatenate(
            [self.xs, self.image_baseline[:, np.newaxis], self.heterogeneity_features],
            axis=1,
        )

        baseline_valid_mask = np.ones((self.xs.shape[0], 1), dtype=np.float32)
        self.feature_valid_mask = np.concatenate(
            [self.feature_valid_mask, baseline_valid_mask, heter_valid_mask.astype(np.float32)],
            axis=1,
        )

    def _build_valid_mask(self):
        self.feature_valid_mask = ~np.isnan(self.xs)
        super()._build_valid_mask()

    def _norm(self):
        super()._norm()
        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)

    def _stage_flat(self):
        super()._stage_flat()
        self.feature_valid_mask = self.feature_valid_mask.reshape(self.H * self.W, -1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int):
        xs = self.xs[idx]
        pos = self.pos[idx]
        valid_mask = self.feature_valid_mask[idx]
        image_baseline = self.image_baseline[idx]
        date = self.date
        return xs, pos, valid_mask, date, image_baseline

    def denorm_y(self, ys: torch.Tensor) -> torch.Tensor:
        y_std = torch.tensor(self.train_dataset.y_std, dtype=ys.dtype, device=ys.device)
        y_mean = torch.tensor(self.train_dataset.y_mean, dtype=ys.dtype, device=ys.device)
        return ys * y_std + y_mean

    def _load_norm_image_baseline(self) -> np.ndarray:
        store = DDPMImageInferenceResultStore(resolution=self.resolution)
        baseline_map = store.get(self.date).astype(np.float32)
        baseline_map = (baseline_map - self.train_dataset.y_mean) / self.train_dataset.y_std
        return baseline_map[self.rows, self.cols]

    def _load_heterogeneity_features(self) -> np.ndarray:
        store = HeterogeneityFeatureStore(resolution=self.resolution)
        heter_map = store.get(self.date).astype(np.float32)
        return heter_map[self.rows, self.cols, :]


class DDPMPixelInferenceResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_INFERENCE_DIR_PATH
        super().__init__(base_dir, resolution)


class DDPMPixelCorrectionResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_CORRECTION_DIR_PATH
        super().__init__(base_dir, resolution)


class HeterogeneityFeatureStore:
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, resolution: str):
        with cls._lock:
            if resolution not in cls._instances:
                cls._instances[resolution] = super().__new__(cls)
        return cls._instances[resolution]

    def __init__(self, resolution: str):
        if getattr(self, "_initialized", False):
            return
        self.resolution = resolution
        self.fine_resolution = RESOLUTION_1KM
        self.coarse_resolution = RESOLUTION_36KM
        self.data_store = ModelDataStore(resolution=self.fine_resolution)
        self.fine_grid_info = GridInfoStore(resolution=self.fine_resolution).get()
        self.coarse_grid_info = GridInfoStore(resolution=self.coarse_resolution).get()
        self.parent_ids, self.expected_counts = build_parent_lookup(
            fine_grid_info=self.fine_grid_info,
            coarse_grid_info=self.coarse_grid_info,
        )
        self.coarse_shape = (self.coarse_grid_info["H"], self.coarse_grid_info["W"])
        self.dem_norm = self._build_dem_heterogeneity()
        self._cache = {}
        self._initialized = True

    def get(self, date: str) -> np.ndarray:
        if date not in self._cache:
            self._cache[date] = self._load(date)
        return self._cache[date]

    def _load(self, date: str) -> np.ndarray:
        ndvi_std = calc_parent_std_map(
            fine_data=self.data_store.get(NDVI_NAME, date),
            parent_ids=self.parent_ids,
            expected_counts=self.expected_counts,
            coarse_shape=self.coarse_shape,
        )
        lst_std = calc_parent_std_map(
            fine_data=self.data_store.get(LST_NAME, date),
            parent_ids=self.parent_ids,
            expected_counts=self.expected_counts,
            coarse_shape=self.coarse_shape,
        )
        coarse_features = np.stack(
            [minmax_norm(ndvi_std), minmax_norm(lst_std), self.dem_norm],
            axis=-1,
        ).astype(np.float32)
        if self.resolution == self.coarse_resolution:
            return coarse_features
        if self.resolution == self.fine_resolution:
            fine_flat = np.full(
                (self.parent_ids.size, coarse_features.shape[-1]),
                np.nan,
                dtype=np.float32,
            )
            flat_parent_ids = self.parent_ids.reshape(-1)
            valid = flat_parent_ids >= 0
            fine_flat[valid] = coarse_features.reshape(-1, coarse_features.shape[-1])[flat_parent_ids[valid]]
            return fine_flat.reshape(self.fine_grid_info["H"], self.fine_grid_info["W"], -1)
        raise ValueError(f"Unsupported heterogeneity feature resolution: {self.resolution}")

    def _build_dem_heterogeneity(self) -> np.ndarray:
        dem = self.data_store.get(DEM_NAME)
        dem_std = calc_parent_std_map(
            fine_data=dem,
            parent_ids=self.parent_ids,
            expected_counts=self.expected_counts,
            coarse_shape=self.coarse_shape,
        )
        return minmax_norm(dem_std)


def build_parent_lookup(fine_grid_info: dict, coarse_grid_info: dict):
    coarse_h = coarse_grid_info["H"]
    coarse_w = coarse_grid_info["W"]
    coarse_ids = np.arange(coarse_h * coarse_w, dtype=np.float32).reshape(coarse_h, coarse_w)
    fine_parent_ids = np.full((fine_grid_info["H"], fine_grid_info["W"]), -1, dtype=np.float32)
    reproject(
        source=coarse_ids,
        destination=fine_parent_ids,
        src_transform=coarse_grid_info["transform"],
        src_crs=coarse_grid_info["crs"],
        dst_transform=fine_grid_info["transform"],
        dst_crs=fine_grid_info["crs"],
        resampling=Resampling.nearest,
    )
    fine_parent_ids = fine_parent_ids.astype(np.int32)
    expected_counts = np.bincount(
        fine_parent_ids.reshape(-1),
        minlength=coarse_h * coarse_w,
    ).astype(np.float32)
    return fine_parent_ids, expected_counts


def calc_parent_std_map(
        fine_data: np.ndarray,
        parent_ids: np.ndarray,
        expected_counts: np.ndarray,
        coarse_shape: Tuple[int, int],
) -> np.ndarray:
    flat_data = fine_data.reshape(-1).astype(np.float64)
    flat_ids = parent_ids.reshape(-1)
    valid = np.isfinite(flat_data) & (flat_ids >= 0)

    coarse_size = coarse_shape[0] * coarse_shape[1]
    valid_counts = np.bincount(flat_ids[valid], minlength=coarse_size).astype(np.float64)
    value_sum = np.bincount(flat_ids[valid], weights=flat_data[valid], minlength=coarse_size)
    value_sq_sum = np.bincount(flat_ids[valid], weights=flat_data[valid] ** 2, minlength=coarse_size)

    mean = np.divide(value_sum, valid_counts, out=np.full(coarse_size, np.nan), where=valid_counts > 0)
    variance = np.divide(
        value_sq_sum,
        valid_counts,
        out=np.full(coarse_size, np.nan),
        where=valid_counts > 0,
    ) - mean ** 2
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)

    coverage = np.divide(
        valid_counts,
        expected_counts,
        out=np.zeros(coarse_size, dtype=np.float64),
        where=expected_counts > 0,
    )
    std[(coverage < MIN_SUBCELL_VALID_RATIO) | (valid_counts < 2)] = np.nan
    return std.reshape(coarse_shape).astype(np.float32)


def minmax_norm(data: np.ndarray) -> np.ndarray:
    out = np.full_like(data, np.nan, dtype=np.float32)
    valid = np.isfinite(data)
    if not np.any(valid):
        return out
    valid_data = data[valid].astype(np.float64)
    data_min = valid_data.min()
    data_max = valid_data.max()
    if np.isclose(data_max, data_min):
        out[valid] = 0.0
        return out
    out[valid] = ((valid_data - data_min) / (data_max - data_min)).astype(np.float32)
    return out
