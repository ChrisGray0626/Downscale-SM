#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM-based Soil Moisture Downscaling Inferencer
@Author Chris
@Date 2025/12/12
"""
from typing import List

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from constants import *
from datasets.dataset import GridInfoStore
from model.ddpm_dataset import DDPMInferenceDataset
from model.module import NoisePredictorImage, build_device
from model.trainer import build_scheduler
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

INFERENCE_STEP_NUM = 250
SM_MIN = 0.02
SM_MAX = 0.5
RESOLUTION = RESOLUTION_1KM


def main():
    device = build_device()
    print(f"Device: {device}")
    model = build_model().to(device)
    grid_info = GridInfoStore(RESOLUTION).get()
    dst_dir_path = os.path.join(INFERENCE_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)
    scheduler = build_scheduler()

    for date in tqdm(get_valid_dates(), desc="Inference"):
        dataset = DDPMInferenceDataset(date=date, resolution=RESOLUTION)
        xs, date_str, insitu_stats = dataset.get_all()
        xs = xs.unsqueeze(0).to(device)
        insitu_stats = torch.from_numpy(insitu_stats).float().unsqueeze(0).to(device)

        pred_y = reverse_diffuse(
            model, scheduler, xs, [date_str], INFERENCE_STEP_NUM, device, insitu_stats=insitu_stats
        )
        pred_y = pred_y.squeeze(0).squeeze(0).cpu().numpy()
        pred_y = dataset.denorm_y(pred_y)
        pred_y = np.clip(pred_y, SM_MIN, SM_MAX)

        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_y, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


def build_model() -> NoisePredictorImage:
    return NoisePredictorImage.from_pretrained(DDPM_MODEL_PATH)


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
    model.eval()
    B, _, H, W = xs.shape
    # Standard normal initial noise
    ys = torch.randn(B, 1, H, W, device=device, dtype=xs.dtype)
    scheduler.set_timesteps(inference_step_num)
    insitu_stats = insitu_stats.to(device)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        # x0-prediction: model outputs x0, scheduler.step expects x0 when prediction_type='sample'
        pred_x0 = model.forward(ys, xs, timesteps, dates=dates, insitu_stats=insitu_stats)
        step_out = scheduler.step(model_output=pred_x0, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


if __name__ == "__main__":
    main()
