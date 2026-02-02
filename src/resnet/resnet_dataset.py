#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet dataset: reuse CommonTrainDataset / CommonInferenceDataset,
only override __getitem__ (and inference _load_data / _filter_valid) for image (5, H, W) → (1, H, W).
@Author Chris
@Date 2026
"""
import numpy as np
import torch

from constants import *
from datasets.common_dataset import CommonInferenceDataset, CommonTrainDataset, FEATURE_NAMES
from utils.data_store import TiffStore

MIN_VALID_RATIO = 0.5


class ResNetTrainDataset(CommonTrainDataset):
    """Only include dates with valid_ratio > MIN_VALID_RATIO."""

    def __init__(self):
        super().__init__()

        # Filter by valid ratio
        n = len(self.dates)
        total = self.valid_masks[0].size
        self._valid_indices = [
            i for i in range(n)
            if self.valid_masks[i].sum() / total > MIN_VALID_RATIO
        ]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]
        x = self.xs[real_idx]  # (H, W, 5), normalized
        y = self.ys[real_idx]  # (H, W), normalized
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        valid = self.valid_masks[real_idx]
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)  # (5, H, W)
        y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32)).unsqueeze(0)  # (1, H, W)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return x, y, valid


class ResNetInferenceDataset(CommonInferenceDataset):

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
        self.xs = xs.astype(np.float32)  # (H, W, 5)

    def _filter_valid(self):
        self.valid = ~np.isnan(self.xs).any(axis=-1)  # (H, W)

    def get_all(self):
        xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)
        xs = torch.from_numpy(xs).permute(2, 0, 1)  # [H, W, 5] -> [5, H, W]

        return xs


class ResNetResultStore(TiffStore):
    def __init__(self, resolution: str):
        base_dir = RESNET_DIR_PATH
        super().__init__(base_dir, resolution)
