#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/2/6
"""
import math
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    """
    seq: [batch_size, ]
    embed_seq: [batch_size, dim]
    """

    def forward(self, seq):
        device = seq.device
        half = self.dim // 2
        embed_seq = math.log(10000) / (half - 1)
        embed_seq = torch.exp(torch.arange(half, device=device) * -embed_seq)  # type: ignore
        embed_seq = seq[:, None].float() * embed_seq[None, :]
        embed_seq = torch.cat([torch.sin(embed_seq), torch.cos(embed_seq)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            embed_seq = F.pad(embed_seq, (0, 1))

        return embed_seq


class TimeEmbedding(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_fourier: int = 8,
            max_doy: int = 366,
            year_min: int = 2016,
            year_max: int = 2020,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.max_doy = max_doy
        self.year_min = float(year_min)
        self.year_max = float(year_max)
        self.doy_dim = 2
        self.year_dim = 1
        self.fourier_dim = 2 * num_fourier
        total_dim = self.doy_dim + self.year_dim + self.fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dates: List[str]) -> torch.Tensor:
        doys, years = [], []
        for date in dates:
            date_obj = datetime.strptime(date, '%Y%m%d')
            doys.append(date_obj.timetuple().tm_yday)
            years.append(date_obj.year)

        device = next(self.proj.parameters()).device
        doy = torch.tensor(doys, dtype=torch.float32, device=device)
        year = torch.tensor(years, dtype=torch.float32, device=device)

        emb_list = []
        doy_norm = doy / self.max_doy
        doy_emb = torch.stack([torch.sin(2 * torch.pi * doy_norm), torch.cos(2 * torch.pi * doy_norm)], dim=1)
        emb_list.append(doy_emb)

        year_norm = (year - self.year_min) / max(self.year_max - self.year_min, 1.0)
        emb_list.append(year_norm.unsqueeze(1))

        base_year = 2016
        time_index = (year - base_year) * 365 + doy
        t = time_index.unsqueeze(1)
        fourier_list = []
        for k in range(self.num_fourier):
            freq = 2 ** k
            fourier_list.append(torch.sin(2 * torch.pi * freq * t / 365))
            fourier_list.append(torch.cos(2 * torch.pi * freq * t / 365))
        emb_list.append(torch.cat(fourier_list, dim=1))

        return self.proj(torch.cat(emb_list, dim=1))


class SpatialEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, num_fourier: int = 6,
                 lon_min: float = -120, lon_max: float = -104,
                 lat_min: float = 35, lat_max: float = 49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.lon_min, self.lon_max = lon_min, lon_max
        self.lat_min, self.lat_max = lat_min, lat_max
        spatial_dim = 2 + 4 * num_fourier
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        lon, lat = pos[:, 0], pos[:, 1]
        lon_n = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
        lat_n = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1

        emb_list = [lon_n.unsqueeze(1), lat_n.unsqueeze(1)]
        for k in range(self.num_fourier):
            freq = 2 ** k
            emb_list.append(torch.sin(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.sin(freq * torch.pi * lat_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lat_n).unsqueeze(1))

        return self.proj(torch.cat(emb_list, dim=1))


class InsituStatsEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, stats_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stats_dim = stats_dim
        self.proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, insitu_stats: torch.Tensor) -> torch.Tensor:
        return self.proj(insitu_stats)


class EarlyStopping:

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def build_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = "mps"

    return device
