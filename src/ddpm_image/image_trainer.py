#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description DDPM_Image-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset

from constants import *
from ddpm_common.module import EarlyStopping, build_device
from ddpm_image.image_dataset import DDPMImageTrainDataset
from ddpm_image.image_module import ImageNoisePredictor

# Diffusion setting
STEP_TOTAL_NUM = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Model setting
INPUT_FEATURE_NUM = 8
HIDDEN_DIM = 512
TIMESTEP_EMB_DIM = 128
RES_BLOCK_NUM = 3

# Train setting
TOTAL_EPOCH = 64
BATCH_SIZE = 8
LR = 1e-4
SEED = 42

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6

# Multi-scale consistency settings.
SCALE_POOL_SIZES = (2, 4)
LAMBDA_SCALE = 0.2
LAMBDA_PATTERN = 0.4


def main():
    seed_everything(SEED)
    dataset = DDPMImageTrainDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=split_generator,
    )

    model = ImageNoisePredictor(
        input_channel_num=INPUT_FEATURE_NUM,
        hidden_dim=HIDDEN_DIM,
        timestep_emb_dim=TIMESTEP_EMB_DIM,
        res_block_num=RES_BLOCK_NUM,
    )

    trainer = Trainer(model, train_dataset, val_dataset)
    model = trainer.run()
    model.save_pretrained(DDPM_IMAGE_MODEL_PATH)


class Trainer:
    def __init__(self, model: ImageNoisePredictor, train_dataset: Dataset, val_dataset: Dataset):
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
        total_valid = 0

        for batch_dates, batch_x, batch_y, batch_valid in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_valid = batch_valid.to(self.device)

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

            pred_y = self.model.forward(
                diffused_ys, batch_x, sampled_timesteps,
                dates=batch_dates,
            )

            loss = compute_loss(
                pred_y=pred_y,
                target_y=batch_y,
                valid=batch_valid,
                timesteps=sampled_timesteps,
                scheduler=self.scheduler,
            )

            valid_sum = batch_valid.sum()
            total_loss += loss.item() * valid_sum.item()
            total_valid += valid_sum.item()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = total_loss / max(total_valid, 1)
        return avg_loss

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
            current_lr = opt.param_groups[0]["lr"]

            self.model.eval()
            val_loss = self.evaluate_epoch(val_loader, optimizer=None)

            print(
                f"Epoch {epoch + 1}/{self.total_epoch} Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f} LR: {current_lr:.6f}"
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
        prediction_type="sample",
        clip_sample=False,
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


def build_early_stopping() -> EarlyStopping:
    return EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
    )


def compute_loss(
        pred_y: torch.Tensor,
        target_y: torch.Tensor,
        valid: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: DDPMScheduler,
) -> torch.Tensor:
    loss_reconstruction = _reconstruction_loss(
        pred_x0=pred_y,
        target_x0=target_y,
        valid=valid,
        timesteps=timesteps,
        scheduler=scheduler,
    )
    loss_scale = _multiscale_consistency_loss(
        pred_x0=pred_y,
        target_x0=target_y,
        valid=valid,
    )
    loss_pattern = _pattern_loss(
        pred_x0=pred_y,
        target_x0=target_y,
        valid=valid,
    )
    return (
            loss_reconstruction
            + LAMBDA_SCALE * loss_scale
            + LAMBDA_PATTERN * loss_pattern
    )


def _reconstruction_loss(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: DDPMScheduler,
) -> torch.Tensor:
    mse_per_pix = (pred_x0 - target_x0) ** 2
    weighted_mse = mse_per_pix * _snr_weights(timesteps, scheduler, pred_x0.dtype)
    valid_sum = valid.sum() + 1e-8
    return (valid * weighted_mse).sum() / valid_sum


def _multiscale_consistency_loss(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
) -> torch.Tensor:
    if len(SCALE_POOL_SIZES) == 0:
        return pred_x0.new_zeros(())

    loss_scale = pred_x0.new_zeros(())
    for pool_size in SCALE_POOL_SIZES:
        pred_lr, valid_lr = _masked_avg_pool(pred_x0, valid, pool_size)
        target_lr, _ = _masked_avg_pool(target_x0, valid, pool_size)
        loss_scale = loss_scale + (((pred_lr - target_lr) ** 2) * valid_lr).sum() / (valid_lr.sum() + 1e-8)
    return loss_scale / float(len(SCALE_POOL_SIZES))


def _pattern_loss(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
) -> torch.Tensor:
    bsz = pred_x0.shape[0]
    pred_flat = pred_x0.reshape(bsz, -1)
    target_flat = target_x0.reshape(bsz, -1)
    valid_flat = valid.reshape(bsz, -1)

    n = valid_flat.sum(dim=1).clamp(min=1.0)
    mean_p = (pred_flat * valid_flat).sum(dim=1) / n
    mean_t = (target_flat * valid_flat).sum(dim=1) / n
    var_p = (valid_flat * (pred_flat - mean_p.unsqueeze(1)) ** 2).sum(dim=1) / n
    var_t = (valid_flat * (target_flat - mean_t.unsqueeze(1)) ** 2).sum(dim=1) / n
    std_p = torch.sqrt(var_p + 1e-8)
    std_t = torch.sqrt(var_t + 1e-8)
    z_p = (pred_flat - mean_p.unsqueeze(1)) / std_p.unsqueeze(1)
    z_t = (target_flat - mean_t.unsqueeze(1)) / std_t.unsqueeze(1)
    loss = ((z_p - z_t) ** 2 * valid_flat).sum(dim=1) / n
    return loss.mean()


def _snr_weights(
        timesteps: torch.Tensor,
        scheduler: DDPMScheduler,
        dtype: torch.dtype,
) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    alpha_bar = alphas_cumprod[timesteps]
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    weights = (snr / (snr + 1.0)).view(-1, 1, 1, 1)
    return weights.to(dtype=dtype)


def _masked_avg_pool(
        x: torch.Tensor,
        valid: torch.Tensor,
        pool_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel = torch.ones(1, 1, pool_size, pool_size, device=x.device, dtype=x.dtype)
    count_valid = F.conv2d(valid, kernel, stride=pool_size)
    pooled_x = F.conv2d(x * valid, kernel, stride=pool_size) / count_valid.clamp(min=1e-8)
    pooled_valid = (count_valid > 0).float()
    return pooled_x, pooled_valid


if __name__ == "__main__":
    main()
