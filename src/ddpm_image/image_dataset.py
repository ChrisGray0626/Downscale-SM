#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM_Image dataset
@Author Chris
@Date 2026
"""
import numpy as np
import torch

from constants import DDPM_IMAGE_INFERENCE_DIR_PATH, DDPM_IMAGE_CORRECTION_DIR_PATH, RANGE
from datasets.base_data_store import BaseTiffStore
from datasets.base_dataset import BaseInferenceDataset, BaseTrainDataset

MIN_VALID_RATIO = 0.2


def _normalize_coords(pos: np.ndarray) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = RANGE
    lon = pos[..., 0]
    lat = pos[..., 1]
    lon_n = 2 * (lon - lon_min) / (lon_max - lon_min) - 1
    lat_n = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    return np.stack([lon_n, lat_n], axis=-1).astype(np.float32)


class DDPMImageTrainDataset(BaseTrainDataset):

    def __init__(self):
        super().__init__(flat=False, filter_valid=False)
        n = len(self.dates)
        total = self.valid_masks[0].size
        self._valid_indices = [
            i for i in range(n)
            if self.valid_masks[i].sum() / total > MIN_VALID_RATIO
        ]
        self._coords = _normalize_coords(self.pos)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]
        date = str(self.dates[real_idx])
        x = self.xs[real_idx]
        y = self.ys[real_idx]
        valid = self.valid_masks[real_idx]

        coords = self._coords
        x_valid_mask = np.isfinite(x).all(axis=-1, keepdims=True).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.concatenate(
            [x.astype(np.float32), coords, x_valid_mask],
            axis=-1,
        )
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
        return date, x, y, valid


class DDPMImageInferenceDataset(BaseInferenceDataset):

    def __init__(self, date: str, resolution: str):
        super().__init__(date, resolution, flat=False, filter_valid=False)
        self._coords = _normalize_coords(self.pos)

    def get_all(self):
        coords = self._coords
        x_valid_mask = np.isfinite(self.xs).all(axis=-1, keepdims=True).astype(np.float32)
        xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)
        xs = np.concatenate(
            [xs.astype(np.float32), coords, x_valid_mask],
            axis=-1,
        )
        xs = torch.from_numpy(xs).permute(2, 0, 1)
        return xs, self.date


class DDPMImageInferenceResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_IMAGE_INFERENCE_DIR_PATH
        super().__init__(base_dir, resolution)


class DDPMImageCorrectionResultStore(BaseTiffStore):

    def __init__(self, resolution: str):
        base_dir = DDPM_IMAGE_CORRECTION_DIR_PATH
        super().__init__(base_dir, resolution)
