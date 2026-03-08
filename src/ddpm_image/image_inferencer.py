#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM_Image-based Soil Moisture Downscaling Inferencer
@Author Chris
@Date 2025/12/12
"""
from typing import List

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from constants import *
from datasets.common_data_store import GridInfoStore
from ddpm_common.module import build_device
from ddpm_image.image_dataset import DDPMImageInferenceDataset
from ddpm_image.image_module import ImageNoisePredictor
from ddpm_image.image_trainer import build_scheduler
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

INFERENCE_STEP_NUM = 50
INFERENCE_SEED = 42
SM_MIN = 0.02
SM_MAX = 0.5
RESOLUTION = RESOLUTION_1KM
APPLY_OUTPUT_CLIP = True


def main():
    device = build_device()
    print(f"Device: {device}")
    model = build_model().to(device)
    grid_info = GridInfoStore(RESOLUTION).get()
    dst_dir_path = os.path.join(DDPM_IMAGE_INFERENCE_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)
    scheduler = build_scheduler()

    for date in tqdm(get_valid_dates(), desc="Inference"):
        dataset = DDPMImageInferenceDataset(date=date, resolution=RESOLUTION)
        pred_y = inference(model=model, dataset=dataset, device=device, scheduler=scheduler)
        pred_y = dataset.denorm_y(pred_y.cpu().numpy())
        if APPLY_OUTPUT_CLIP:
            pred_y = np.clip(pred_y, SM_MIN, SM_MAX)

        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_y, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


def build_model() -> ImageNoisePredictor:
    return ImageNoisePredictor.from_pretrained(DDPM_IMAGE_MODEL_PATH)


@torch.no_grad()
def inference(
        model: ImageNoisePredictor,
        dataset: DDPMImageInferenceDataset,
        device: str,
        scheduler: DDPMScheduler,
) -> torch.Tensor:
    model = model.to(device)
    model.eval()

    xs, date_str = dataset.get_all()
    xs = xs.unsqueeze(0).to(device)
    dates = [date_str]

    pred_y = reverse_diffuse(
        model, scheduler, xs, dates, INFERENCE_STEP_NUM, device, INFERENCE_SEED
    )
    pred_y = pred_y.squeeze(0).squeeze(0)
    return pred_y


@torch.no_grad()
def reverse_diffuse(
        model: ImageNoisePredictor,
        scheduler: DDPMScheduler,
        xs: torch.Tensor,
        dates: List[str],
        inference_step_num: int,
        device: str,
        seed: int,
) -> torch.Tensor:
    model.eval()
    B, _, H, W = xs.shape
    generator = torch.Generator().manual_seed(seed)
    ys = torch.randn(B, 1, H, W, dtype=xs.dtype, generator=generator).to(device)
    scheduler.set_timesteps(inference_step_num)
    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        pred_x0 = model.forward(ys, xs, timesteps, dates=dates)
        step_out = scheduler.step(model_output=pred_x0, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


if __name__ == "__main__":
    main()
