#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description DDPM_Image-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
from typing import List

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
INPUT_FEATURE_NUM = 5
HIDDEN_DIM = 512
TIMESTEP_EMB_DIM = 128
RES_BLOCK_NUM = 3

# Train setting
TOTAL_EPOCH = 64
BATCH_SIZE = 8
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6

# Scale consistency: multi-scale masked avg_pool(pred_x0) vs avg_pool(target_x0); weight for loss_scale
# Use both 2x2 and 4x4 local blocks for low-res comparison
SCALE_POOL_SIZES = (2, 4)
LAMBDA_SCALE = 0.6
# Variance consistency (on valid pixels): weight for loss_var
LAMBDA_VAR = 0


# TODO Loss
def diffusion_loss_x0(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: DDPMScheduler,
) -> torch.Tensor:
    mse_per_pix = (pred_x0 - target_x0) ** 2

    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    alpha_bar = alphas_cumprod[timesteps]  # [B]
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    weights = snr / (snr + 1.0)
    weights = weights.view(-1, 1, 1, 1)

    weighted_mse = mse_per_pix * weights
    valid_sum = valid.sum() + 1e-8
    return (valid * weighted_mse).sum() / valid_sum


def _scale_loss_valid_only(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
        pool_size: int,
) -> torch.Tensor:
    """
    Scale consistency only over valid region: masked avg_pool (average only over valid
    pixels in each window), then MSE only on low-res cells that have at least one valid pixel.
    """
    k = pool_size
    ones = torch.ones(1, 1, k, k, device=pred_x0.device, dtype=pred_x0.dtype)

    count_v = F.conv2d(valid, ones, stride=k)
    sum_p = F.conv2d(pred_x0 * valid, ones, stride=k)
    sum_t = F.conv2d(target_x0 * valid, ones, stride=k)

    pred_lr = sum_p / count_v.clamp(min=1e-8)
    target_lr = sum_t / count_v.clamp(min=1e-8)
    valid_lr = (count_v > 0).float()

    loss_scale = ((pred_lr - target_lr) ** 2 * valid_lr).sum() / (valid_lr.sum() + 1e-8)
    return loss_scale


def _variance_consistency_loss(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
) -> torch.Tensor:
    """
    Variance consistency on valid pixels (per-sample): encourage Var(pred_x0) ~= Var(target_x0).
    This directly targets the ddpm_common issue slope < 1 (variance shrinkage).
    """
    B = pred_x0.shape[0]
    pred_flat = pred_x0.reshape(B, -1)
    target_flat = target_x0.reshape(B, -1)
    valid_flat = valid.reshape(B, -1)

    n = valid_flat.sum(dim=1).clamp(min=1.0)  # [B]
    mean_p = (pred_flat * valid_flat).sum(dim=1) / n
    mean_t = (target_flat * valid_flat).sum(dim=1) / n

    var_p = (valid_flat * (pred_flat - mean_p.unsqueeze(1)) ** 2).sum(dim=1) / n
    var_t = (valid_flat * (target_flat - mean_t.unsqueeze(1)) ** 2).sum(dim=1) / n

    return ((var_p - var_t) ** 2).mean()


def combined_loss(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        valid: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: DDPMScheduler,
) -> torch.Tensor:
    """
    loss_x0 = SNR-weighted MSE (diffusion_loss_x0)
    loss_scale = MSE between masked avg_pool(pred) and masked avg_pool(target), only on valid low-res cells.
    loss = loss_x0 + lambda_scale * loss_scale
    Used for both training and validation.
    """
    loss_x0 = diffusion_loss_x0(
        pred_x0=pred_x0,
        target_x0=target_x0,
        valid=valid,
        timesteps=timesteps,
        scheduler=scheduler,
    )

    # Multi-scale scale consistency: average losses over all pool sizes
    loss_scale = 0.0
    for k in SCALE_POOL_SIZES:
        loss_scale = loss_scale + _scale_loss_valid_only(
            pred_x0=pred_x0,
            target_x0=target_x0,
            valid=valid,
            pool_size=k,
        )
    loss_scale = loss_scale / float(len(SCALE_POOL_SIZES))

    loss_var = _variance_consistency_loss(pred_x0=pred_x0, target_x0=target_x0, valid=valid)

    return loss_x0 + LAMBDA_SCALE * loss_scale + LAMBDA_VAR * loss_var


def main():
    dataset = DDPMImageTrainDataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
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
        self.device = build_device()
        self.early_stopping = build_early_stopping()
        self.model = model.to(self.device)

    def evaluate_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer = None) -> float:
        total_loss = 0.0
        total_valid = 0

        for batch_dates, batch_x, batch_y, batch_valid, batch_insitu_stats in data_loader:
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

            # x0-prediction: ddpm_image predicts clean sample x0 ≈ batch_y
            pred_x0 = self.model.forward(
                diffused_ys, batch_x, sampled_timesteps,
                dates=batch_dates,
                insitu_stats=batch_insitu_stats,
            )

            # Same combined loss for train and val: loss_x0 (SNR-weighted MSE) + scale (avg_pool vs target_lr)
            loss = combined_loss(
                pred_x0=pred_x0,
                target_x0=batch_y,
                valid=batch_valid,
                timesteps=sampled_timesteps,
                scheduler=self.scheduler,
            )

            # 累加时使用当前 batch 的有效像素数做加权
            valid_sum = batch_valid.sum()
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
        # x0-prediction mode: ddpm_image outputs x0, not noise epsilon
        prediction_type="sample",
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
        model: ImageNoisePredictor,
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
        # x0-prediction: ddpm_image outputs x0, scheduler.step expects x0 when prediction_type='sample'
        pred_x0 = model.forward(ys, xs, timesteps, dates=dates, insitu_stats=insitu_stats)
        step_out = scheduler.step(model_output=pred_x0, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


if __name__ == "__main__":
    main()

