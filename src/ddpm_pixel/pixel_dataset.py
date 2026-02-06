#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from typing import Tuple

import numpy as np
import torch

from constants import RESOLUTION_36KM, DDPM_PIXEL_INFERENCE_DIR_PATH, DDPM_PIXEL_CORRECTION_DIR_PATH
from datasets.base_data_store import BaseTiffStore
from datasets.base_dataset import BaseInferenceDataset, BaseTrainDataset
from datasets.common_data_store import InsituStatsStore


class DDPMPixelTrainDataset(BaseTrainDataset):

    def __init__(self):
        # Keep all pixels after normalization/flattening; we'll handle NaNs ourselves.
        super().__init__(flat=True, filter_valid=False)
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)

        # xs, ys are normalized at this point.
        # Build per-feature valid mask BEFORE imputation: True where original xs was not NaN.
        feature_valid_mask = ~np.isnan(self.xs)  # (N, F)

        # Only keep samples with valid target y (we don't want to learn on NaN labels).
        valid_y = ~np.isnan(self.ys)  # (N,)

        self.xs_valid_mask = feature_valid_mask[valid_y].astype(np.float32)
        self.xs = self.xs[valid_y]
        self.ys = self.ys[valid_y]
        self.dates = self.dates[valid_y]
        self.pos = self.pos[valid_y]
        self.rows = self.rows[valid_y]
        self.cols = self.cols[valid_y]

        # In normalized feature space, impute any remaining NaNs to 0 (= per-channel mean level).
        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int) -> Tuple:
        date = str(self.dates[idx])
        pos = self.pos[idx].astype(np.float32)
        xs = self.xs[idx].astype(np.float32)
        ys = np.asarray(self.ys[idx], dtype=np.float32)
        insitu_stats = self.insitu_stats_store.get(date)
        valid_mask = self.xs_valid_mask[idx].astype(np.float32)

        xs_t = torch.from_numpy(xs)
        ys_t = torch.tensor(ys, dtype=torch.float32)
        pos_t = torch.from_numpy(pos)
        insitu_stats_t = torch.from_numpy(insitu_stats.astype(np.float32))
        valid_mask_t = torch.from_numpy(valid_mask)

        # Order is aligned with Trainer loop: (dates, pos, xs, ys, insitu_stats, valid_mask)
        return date, pos_t, xs_t, ys_t, insitu_stats_t, valid_mask_t


class DDPMPixelInferenceDataset(BaseInferenceDataset):

    def __init__(self, date: str, resolution: str = RESOLUTION_36KM):
        self.date = date
        self.resolution = resolution
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)
        self.insitu_stats = self.insitu_stats_store.get(self.date)

        # For DDPM-based inference we want a value for every pixel:
        # - do NOT drop invalid pixels (filter_valid=False)
        # - instead, work in normalized feature space and impute NaNs to 0 (mean level),
        #   and expose a per-feature valid mask to the model.
        super().__init__(date=self.date, resolution=self.resolution, flat=True, filter_valid=False)

    def _build_valid_mask(self):
        # xs is still in original (unnormalized) space here.
        # Build per-feature valid mask BEFORE normalization: True where original xs was not NaN.
        self.feature_valid_mask = ~np.isnan(self.xs)  # (H, W, F)
        super()._build_valid_mask()

    def _norm(self):
        # First run the standard normalization using train statistics
        super()._norm()
        # Then, in normalized space, replace any remaining NaNs / infs with 0,
        # which corresponds to the per-channel mean in the original space.
        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)

    def _stage_flat(self):
        super()._stage_flat()
        # Flatten feature_valid_mask in the same way as xs/pos/rows/cols.
        self.feature_valid_mask = self.feature_valid_mask.reshape(self.H * self.W, -1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int):
        xs = self.xs[idx]
        pos = self.pos[idx]
        valid_mask = self.feature_valid_mask[idx]
        date = self.date
        return xs, pos, valid_mask, date

    def denorm_y(self, ys: torch.Tensor) -> torch.Tensor:
        y_std = torch.tensor(self.train_dataset.y_std, dtype=ys.dtype, device=ys.device)
        y_mean = torch.tensor(self.train_dataset.y_mean, dtype=ys.dtype, device=ys.device)
        return ys * y_std + y_mean


class DDPMPixelInferenceResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_INFERENCE_DIR_PATH
        super().__init__(base_dir, resolution)


class DDPMPixelCorrectionResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_PIXEL_CORRECTION_DIR_PATH
        super().__init__(base_dir, resolution)
