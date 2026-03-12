#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Pixel DDPM dataset with DDPM_Image-conditioned residual targets
  @Author Chris
  @Date 2026/3/9
"""

from typing import Tuple

import numpy as np
import torch

from constants import RESOLUTION_36KM, DDPM_PIXEL_INFERENCE_DIR_PATH, DDPM_PIXEL_CORRECTION_DIR_PATH
from datasets.base_data_store import BaseTiffStore
from datasets.base_dataset import BaseInferenceDataset, BaseTrainDataset
from ddpm_image.image_dataset import DDPMImageInferenceResultStore


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
        self.residuals = self.ys - self.image_baselines

        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)
        self.image_baselines = np.nan_to_num(self.image_baselines, nan=0.0, posinf=0.0, neginf=0.0)
        self.residuals = np.nan_to_num(self.residuals, nan=0.0, posinf=0.0, neginf=0.0)
        self.xs = np.concatenate([self.xs, self.image_baselines[:, np.newaxis]], axis=1)

        baseline_valid_mask = np.ones((self.xs.shape[0], 1), dtype=np.float32)
        self.xs_valid_mask = np.concatenate([self.xs_valid_mask, baseline_valid_mask], axis=1)

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


class DDPMPixelInferenceDataset(BaseInferenceDataset):

    def __init__(self, date: str, resolution: str = RESOLUTION_36KM):
        self.date = date
        self.resolution = resolution
        super().__init__(date=self.date, resolution=self.resolution, flat=True, filter_valid=False)
        self.image_baseline = self._load_norm_image_baseline()
        self.image_baseline = np.nan_to_num(self.image_baseline, nan=0.0, posinf=0.0, neginf=0.0)
        self.xs = np.concatenate([self.xs, self.image_baseline[:, np.newaxis]], axis=1)

        baseline_valid_mask = np.ones((self.xs.shape[0], 1), dtype=np.float32)
        self.feature_valid_mask = np.concatenate([self.feature_valid_mask, baseline_valid_mask], axis=1)

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


class DDPMPixelInferenceResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_INFERENCE_DIR_PATH
        super().__init__(base_dir, resolution)


class DDPMPixelCorrectionResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_CORRECTION_DIR_PATH
        super().__init__(base_dir, resolution)
