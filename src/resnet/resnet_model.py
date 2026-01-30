#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet for image-to-image: 5-channel input → 1-channel output.
Style: BaseResBlock + GroupNorm + SiLU; save/load config + state_dict + norm stats.
@Author Chris
@Date 2026
"""
import torch
import torch.nn as nn

DEFAULT_CONFIG = dict(in_channels=5, out_channels=1, base_channels=512, num_blocks=3)


class BaseResBlock(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResBlock(BaseResBlock):

    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        num_groups = min(8, hidden_dim)
        if hidden_dim % num_groups != 0:
            num_groups = 1
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.SiLU(),
        )


class ResNet(nn.Module):

    def __init__(
            self,
            in_channels: int = 5,
            out_channels: int = 1,
            base_channels: int = 64,
            num_blocks: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_blocks)]
        )

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W), e.g. (B, 5, H, W)
        Returns:
            (B, out_channels, H, W), e.g. (B, 1, H, W)
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def save_checkpoint(model: ResNet, x_mean, x_std, y_mean, y_std, path: str):
    """保存：config + state_dict + 归一化统计量。"""
    config = dict(
        in_channels=model.in_channels,
        out_channels=model.out_channels,
        base_channels=model.stem[0].out_channels,
        num_blocks=len(model.blocks),
    )
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, path)


def load_checkpoint(path: str, device=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = ckpt["config"]
    else:
        sd = ckpt["model_state_dict"]
        config = {
            **DEFAULT_CONFIG,
            "base_channels": int(sd["stem.0.weight"].shape[0]),
            "num_blocks": sum(1 for k in sd if k.startswith("blocks.") and ".net.0.weight" in k),
        }
    model = ResNet(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model = model.to(device)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])
    return model, y_mean, y_std
