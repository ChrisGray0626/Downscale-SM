#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Module for Task
  @Author Chris
  @Date 2025/11/12
"""
from typing import List

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ddpm_common.module import SinusoidalPosEmb, TimeEmbedding


class ImageNoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            input_channel_num: int = 8,
            output_channel_num: int = 1,
            hidden_dim: int = 512,
            timestep_emb_dim: int = 128,
            res_block_num: int = 3,
            dropout_p: float = 0.1,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channel_num + output_channel_num, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU(),
        )

        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, hidden_dim),
        )
        self.time_embedding = TimeEmbedding(hidden_dim=hidden_dim, num_fourier=8)
        condition_dim = hidden_dim * 2
        self.condition_fusion = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        dilation_cycle = (1, 2, 4)
        self.net = nn.ModuleList([
            FiLMResBlock2D(
                hidden_dim,
                hidden_dim,
                dropout_p=dropout_p,
                dilation=dilation_cycle[i % len(dilation_cycle)],
            )
            for i in range(res_block_num)
        ])
        self.head = nn.Conv2d(hidden_dim, output_channel_num, kernel_size=3, padding=1)

    def forward(
            self,
            diffused_ys: torch.Tensor,
            xs: torch.Tensor,
            timesteps: torch.Tensor,
            dates: List[str],
    ) -> torch.Tensor:
        x = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(x)

        embed_timestep = self.timestep_embedding(timesteps)
        embed_time = self.time_embedding(dates)
        condition = torch.cat([embed_timestep, embed_time], dim=1)
        condition = self.condition_fusion(condition)

        for layer in self.net:
            x = layer(x, condition)

        return self.head(x)


class FiLMResBlock2D(nn.Module):

    def __init__(
            self,
            channels: int,
            condition_dim: int,
            dropout_p: float = 0.1,
            dilation: int = 1,
    ):
        super().__init__()
        self.film = FiLM2D(condition_dim, out_channels=channels)
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        branch = self.net(self.film(x, condition))
        return x + branch


class FiLM2D(nn.Module):

    def __init__(self, condition_dim: int, out_channels: int = None):
        super().__init__()
        self.out_channels = out_channels or condition_dim
        self.scale_net = nn.Sequential(
            nn.Linear(condition_dim, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(condition_dim, self.out_channels),
            nn.SiLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        if x.dim() == 4:
            B = x.shape[0]
            scale = scale.view(B, -1, 1, 1)
            shift = shift.view(B, -1, 1, 1)
        return scale * x + shift
