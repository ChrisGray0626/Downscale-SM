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
        super().__init__(flat=True, filter_valid=True)
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int) -> Tuple:
        date = str(self.dates[idx])
        pos = self.pos[idx].astype(np.float32)
        xs = self.xs[idx].astype(np.float32)
        ys = np.asarray(self.ys[idx], dtype=np.float32)
        insitu_stats = self.insitu_stats_store.get(date)

        xs = torch.from_numpy(xs)
        ys = torch.tensor(ys, dtype=torch.float32)
        pos = torch.from_numpy(pos)
        insitu_stats = torch.from_numpy(insitu_stats)

        return date, pos, xs, ys, insitu_stats


class DDPMPixelInferenceDataset(BaseInferenceDataset):

    def __init__(self, date: str, resolution: str = RESOLUTION_36KM):
        self.date = date
        self.resolution = resolution
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)
        self.insitu_stats = self.insitu_stats_store.get(self.date)

        super().__init__(date=self.date, resolution=self.resolution, flat=True, filter_valid=True)

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(self, idx: int):
        xs = self.xs[idx]
        pos = self.pos[idx]
        date = self.date
        return xs, pos, date

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
