#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Residual DDPM_Pixel trainer conditioned on DDPM_Image outputs
  @Author Chris
  @Date 2026/3/9
"""
import random

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from constants import *
from ddpm_common.module import build_device, EarlyStopping
from ddpm_pixel.pixel_dataset import DDPMPixelTrainDataset
from ddpm_pixel.pixel_module import PixelNoisePredictor

# Dataset setting
INPUT_FEATURE_NUM = 6

# Diffusion setting
STEP_TOTAL_NUM = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Model setting
HIDDEN_DIM = 512
TIMESTEP_EMB_DIM = 128
RES_BLOCK_NUM = 3

# Train setting
TOTAL_EPOCH = 60
BATCH_SIZE = 64
LR = 1e-4
SEED = 42

# Loss setting
LAMBDA_RESIDUAL_L1 = 0.05

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6


def main():
    seed_everything(SEED)
    dataset = DDPMPixelTrainDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    model = PixelNoisePredictor(
        input_feature_num=INPUT_FEATURE_NUM,
        hidden_dim=HIDDEN_DIM,
        timestep_emb_dim=TIMESTEP_EMB_DIM,
        res_block_num=RES_BLOCK_NUM,
    )

    trainer = Trainer(model, train_dataset, val_dataset)
    model = trainer.run()
    model.save_pretrained(DDPM_PIXEL_MODEL_PATH)


class Trainer:

    def __init__(self, model: PixelNoisePredictor,
                 train_dataset: Dataset, val_dataset: Dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler = build_scheduler()
        self.total_epoch = TOTAL_EPOCH
        self.batch_size = BATCH_SIZE
        self.lr = LR
        self.seed = SEED
        self.device = build_device()
        self.early_stopping = build_early_stopping()
        self.model = model.to(self.device)

    def evaluate_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer = None) -> float:
        total_loss = 0.0
        total_samples = 0

        for batch_dates, batch_pos, batch_xs, batch_residuals, batch_valid_mask in data_loader:
            batch_pos = batch_pos.to(self.device)
            batch_xs = batch_xs.to(self.device)
            batch_residuals = batch_residuals.to(self.device).unsqueeze(1)
            batch_valid_mask = batch_valid_mask.to(self.device)

            batch_size = batch_xs.shape[0]
            sampled_timesteps = torch.randint(
                0, self.scheduler.config['num_train_timesteps'],
                (batch_size,), device=self.device, dtype=torch.long
            )

            noises = torch.randn_like(batch_residuals)
            diffused_residuals = self.scheduler.add_noise(
                original_samples=batch_residuals,
                noise=noises,
                timesteps=sampled_timesteps  # type: ignore[arg-type]
            )

            pred_residuals = self.model.forward(
                diffused_residuals, batch_xs, sampled_timesteps,
                pos=batch_pos, dates=batch_dates,
                valid_mask=batch_valid_mask,
            )

            loss = calc_loss(
                pred_residuals=pred_residuals,
                true_residuals=batch_residuals,
                timesteps=sampled_timesteps,
                scheduler=self.scheduler,
            )

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return total_loss / max(total_samples, 1)

    def run(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_epoch, eta_min=1e-6
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        for epoch in range(self.total_epoch):
            self.model.train()
            train_loss = self.evaluate_epoch(train_loader, optimizer=opt)
            current_lr = opt.param_groups[0]['lr']

            self.model.eval()
            val_loss = self.evaluate_epoch(val_loader, optimizer=None)

            print(
                f"Epoch {epoch + 1}/{self.total_epoch} Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f} LR: {current_lr:.6f}"
            )

            if self.early_stopping(val_loss, self.model):
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best val loss: {self.early_stopping.best_loss:.6f}"
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
        prediction_type="sample",
        clip_sample=False,
    )


def build_early_stopping() -> EarlyStopping:
    return EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calc_loss(pred_residuals: torch.Tensor, true_residuals: torch.Tensor,
              timesteps: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
    mse = (pred_residuals - true_residuals) ** 2
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    alpha_bar = alphas_cumprod[timesteps]
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    weights = (snr / (snr + 1.0)).view(-1, 1)
    loss_reconstruction = (weights * mse).mean()
    loss_small_residual = pred_residuals.abs().mean()
    return loss_reconstruction + LAMBDA_RESIDUAL_L1 * loss_small_residual


if __name__ == "__main__":
    main()
