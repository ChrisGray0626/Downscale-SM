#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM dataset
@Author Chris
@Date 2026
"""
import numpy as np
import torch

from datasets.common_dataset import CommonInferenceDataset, CommonTrainDataset
from datasets.dataset import InsituStatsStore

MIN_VALID_RATIO = 0.5


class DDPMTrainDataset(CommonTrainDataset):

    def __init__(self):
        super().__init__(flat=False, filter_valid=False)
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)
        n = len(self.dates)
        total = self.valid_masks[0].size
        self._valid_indices = [
            i for i in range(n)
            if self.valid_masks[i].sum() / total > MIN_VALID_RATIO
        ]
        self._insitu_stats_list = [
            self.insitu_stats_store.get(str(self.dates[i]))
            for i in self._valid_indices
        ]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self._valid_indices[idx]
        date = str(self.dates[real_idx])
        x = self.xs[real_idx]
        y = self.ys[real_idx]
        valid = self.valid_masks[real_idx]
        insitu_stats = self._insitu_stats_list[idx].astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
        insitu_stats = torch.from_numpy(insitu_stats)
        return date, x, y, valid, insitu_stats


class DDPMInferenceDataset(CommonInferenceDataset):

    def __init__(self, date: str, resolution: str):
        super().__init__(date, resolution, flat=False, filter_valid=False)
        self.insitu_stats_store = InsituStatsStore(resolution=resolution)
        self.insitu_stats = self.insitu_stats_store.get(date).astype(np.float32)

    # def _load_data(self):
    #     grid_info = self.grid_info_store.get()
    #     self.H, self.W = grid_info["H"], grid_info["W"]
    #     self.grid_info = grid_info
    #     self.pos = np.asarray(grid_info["pos"], dtype=np.float64)
    #     self.rows_full = grid_info["rows"].flatten()
    #     self.cols_full = grid_info["cols"].flatten()
    #     xs = np.stack(
    #         [self.data_store.get(name, self.date) for name in FEATURE_NAMES],
    #         axis=-1,
    #     )
    #     self.xs = xs.astype(np.float32)

    def get_all(self):
        xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)
        xs = torch.from_numpy(xs).permute(2, 0, 1)
        return xs, self.date, self.insitu_stats
