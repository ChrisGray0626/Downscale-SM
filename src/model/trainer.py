#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description DDPM-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
from typing import List

import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset

from constants import *
from model.ddpm_dataset import DDPMTrainDataset
from model.module import EarlyStopping, NoisePredictorImage, build_device

# Diffusion setting
STEP_TOTAL_NUM = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Model setting
INPUT_FEATURE_NUM = 5
HIDDEN_DIM = 512
TIMESTEP_EMB_DIM = 128
RES_BLOCK_NUM = 3

# Train setting
TOTAL_EPOCH = 60
BATCH_SIZE = 8
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6

# Charbonnier loss
CHARBONNIER_EPS = 1e-3


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = CHARBONNIER_EPS) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps) - eps


def main():
    dataset = DDPMTrainDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    model = NoisePredictorImage(
        input_feature_num=INPUT_FEATURE_NUM,
        hidden_dim=HIDDEN_DIM,
        timestep_emb_dim=TIMESTEP_EMB_DIM,
        res_block_num=RES_BLOCK_NUM,
    )

    trainer = Trainer(model, train_dataset, val_dataset)
    model = trainer.run()
    model.save_pretrained(DDPM_MODEL_PATH)


class Trainer:
    def __init__(self, model: NoisePredictorImage, train_dataset: Dataset, val_dataset: Dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler = build_scheduler()
        self.total_epoch = TOTAL_EPOCH
        self.batch_size = BATCH_SIZE
        self.lr = LR
        self.device = build_device()
        self.early_stopping = build_early_stopping()
        self.model = model.to(self.device)

    def evaluate_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer = None) -> float:
        total_loss = 0.0
        total_valid = 0

        for batch_x, batch_y, batch_valid, batch_dates, batch_insitu_stats in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_valid = batch_valid.to(self.device)
            batch_insitu_stats = batch_insitu_stats.to(self.device)

            B = batch_x.shape[0]
            sampled_timesteps = torch.randint(
                0,
                self.scheduler.config["num_train_timesteps"],
                (B,),
                device=self.device,
                dtype=torch.long,
            )

            noises = torch.randn_like(batch_y)
            diffused_ys = self.scheduler.add_noise(
                original_samples=batch_y,
                noise=noises,
                timesteps=sampled_timesteps,  # type: ignore
            )

            pred_noise = self.model.forward(
                diffused_ys, batch_x, sampled_timesteps,
                dates=list(batch_dates),
                insitu_stats=batch_insitu_stats,
            )

            loss_per_pix = charbonnier_loss(pred_noise, noises)
            valid_sum = batch_valid.sum() + 1e-8
            loss = (batch_valid * loss_per_pix).sum() / valid_sum

            total_loss += loss.item() * valid_sum.item()
            total_valid += valid_sum.item()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return total_loss / max(total_valid, 1)

    def run(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_epoch, eta_min=1e-6
        )

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.total_epoch):
            self.model.train()
            train_loss = self.evaluate_epoch(train_loader, optimizer=opt)
            current_lr = opt.param_groups[0]["lr"]

            self.model.eval()
            val_loss = self.evaluate_epoch(val_loader, optimizer=None)

            print(
                f"Epoch {epoch + 1}/{self.total_epoch} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f} LR: {current_lr:.6f}"
            )

            if self.early_stopping(val_loss, self.model):
                print(
                    f"Early stopping at epoch {epoch + 1}. Best val loss: {self.early_stopping.best_loss:.6f}"
                )
                break
            lr_scheduler.step()

        return self.model


def build_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=STEP_TOTAL_NUM,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


def build_early_stopping() -> EarlyStopping:
    return EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
    )


@torch.no_grad()
def reverse_diffuse(
        model: NoisePredictorImage,
        scheduler: DDPMScheduler,
        xs: torch.Tensor,
        dates: List[str],
        inference_step_num: int,
        device: str,
        insitu_stats: torch.Tensor,
) -> torch.Tensor:
    """Reverse diffusion for full image: xs (B, 5, H, W) -> ys (B, 1, H, W)."""
    model.eval()
    B, _, H, W = xs.shape
    ys = torch.randn(B, 1, H, W, device=device, dtype=xs.dtype)
    scheduler.set_timesteps(inference_step_num)
    insitu_stats = insitu_stats.to(device)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        pred_noises = model.forward(ys, xs, timesteps, dates=dates, insitu_stats=insitu_stats)
        step_out = scheduler.step(model_output=pred_noises, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


if __name__ == "__main__":
    main()
