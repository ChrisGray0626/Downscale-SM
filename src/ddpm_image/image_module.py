#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Module for Task
  @Author Chris
  @Date 2025/11/12
"""
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ddpm_common.module import SinusoidalPosEmb, TimeEmbedding, InsituStatsEmbedding


class ImageNoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            input_channel_num: int = 5,
            output_channel_num: int = 1,
            hidden_dim: int = 512,
            timestep_emb_dim: int = 128,
            res_block_num: int = 3,
            channel_attention_reduction: int = 16,
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
        self.insitu_stats_embedding = InsituStatsEmbedding(hidden_dim=hidden_dim, stats_dim=4)
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.net = nn.ModuleList([
            FiLMResBlock2D(
                hidden_dim,
                hidden_dim,
                channel_attention_reduction=channel_attention_reduction,
                dropout_p=dropout_p,
            )
            for _ in range(res_block_num)
        ])
        self.head = nn.Conv2d(hidden_dim, output_channel_num, kernel_size=3, padding=1)

    def forward(
            self,
            diffused_ys: torch.Tensor,
            xs: torch.Tensor,
            timesteps: torch.Tensor,
            dates: List[str],
            insitu_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(x)

        embed_timestep = self.timestep_embedding(timesteps)
        embed_time = self.time_embedding(dates)
        embed_insitu_stats = self.insitu_stats_embedding(insitu_stats)
        condition = torch.cat([embed_timestep, embed_time, embed_insitu_stats], dim=1)
        condition = self.condition_fusion(condition)

        for layer in self.net:
            x = layer(x, condition)

        return self.head(x)


class ChannelAttention(nn.Module):

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_dim = max(dim // reduction, 1)
        self.MLP = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        avg_out = self.MLP(avg_out)
        max_out = self.MLP(max_out)
        out = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * out


class FiLMResBlock2D(nn.Module):

    def __init__(
            self,
            channels: int,
            condition_dim: int,
            channel_attention_reduction: int = 16,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        self.film = FiLM2D(condition_dim, out_channels=channels)
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.channel_attn = ChannelAttention(dim=channels, reduction=channel_attention_reduction)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        branch = self.net(self.film(x, condition))
        branch = self.channel_attn(branch)
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
