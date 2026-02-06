#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/2/6
"""
from typing import List

import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from torch import nn

from constants import RANGE
from ddpm_common.module import SinusoidalPosEmb, TimeEmbedding, SpatialEmbedding, InsituStatsEmbedding


class PixelNoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_feature_num: int,
                 hidden_dim: int = 512, timestep_emb_dim: int = 128,
                 res_block_num: int = 3):
        super().__init__()
        self.input_feature_num = input_feature_num
        self.hidden_dim = hidden_dim

        # Use both normalized features and their valid mask as inputs, plus diffused y.
        # inputs = [xs (F), diffused_y (1), valid_mask (F)] -> total 2F + 1.
        input_dim = input_feature_num * 2 + 1
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Timestep Embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.emb_timestep2hidden = nn.Linear(timestep_emb_dim * 4, hidden_dim)

        # Time Embedding
        self.time_embedding = TimeEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=8
        )

        # Spatial Embedding
        lon_min, lat_min, lon_max, lat_max = RANGE
        self.spatial_embedding = SpatialEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=6,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )

        # Insitu Stats Embedding
        self.insitu_stats_embedding = InsituStatsEmbedding(
            hidden_dim=hidden_dim,
            stats_dim=4
        )

        # Condition Fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Residual Block
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim=hidden_dim)
            for _ in range(res_block_num)
        ])

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor,
                pos: torch.Tensor, dates: List[str],
                insitu_stats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # Concatenate normalized features, current diffused y, and per-feature valid mask.
        inputs = torch.cat([xs, diffused_ys, valid_mask], dim=1)
        x = self.input_layer(inputs)

        # Embed Timestep
        embed_timesteps = self.timestep_embedding(timesteps)
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # Embed Time
        embed_time = self.time_embedding(dates)

        # Embed Spatial
        embed_spatial = self.spatial_embedding(pos)

        # Embed Insitu Stats
        embed_insitu_stats = self.insitu_stats_embedding(insitu_stats)

        # Fuse Condition
        condition = torch.cat([embed_timesteps, embed_time, embed_spatial, embed_insitu_stats], dim=1)
        condition = self.condition_fusion(condition)

        # Residual Blocks with FiLM
        for res_block in self.res_blocks:
            x = res_block(x, condition)

        out = self.output_layer(x)

        return out


class FiLMResBlock(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.film = FiLM(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_modulated = self.film(x, condition)
        out = x + self.net(x_modulated)

        return out


class FiLM(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        out = scale * x + shift

        return out
