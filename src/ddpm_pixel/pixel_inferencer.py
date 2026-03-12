#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Residual DDPM_Pixel inferencer conditioned on DDPM_Image outputs
  @Author Chris
  @Date 2026/3/9
"""
from datetime import datetime
from typing import List

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import *
from datasets.common_data_store import GridInfoStore
from ddpm_common.module import build_device
from ddpm_pixel.pixel_dataset import DDPMPixelInferenceDataset
from ddpm_pixel.pixel_module import PixelNoisePredictor
from ddpm_pixel.pixel_trainer import build_scheduler
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

INFERENCE_STEP_NUM = 50
BASE_INFERENCE_SEED = 42
BATCH_SIZE = 16384

SM_MIN = 0.02
SM_MAX = 0.5

RESOLUTION = RESOLUTION_36KM


def main():
    device = build_device()
    print(f"Device: {device}")
    model = build_model()
    grid_info = GridInfoStore(RESOLUTION).get()
    dst_dir_path = os.path.join(DDPM_PIXEL_INFERENCE_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)

    for date in tqdm(get_valid_dates(), desc="Inference"):
        dataset = DDPMPixelInferenceDataset(date=date, resolution=RESOLUTION)
        pred_ys = inference(model=model, dataset=dataset, device=device)
        pred_ys = dataset.denorm_y(pred_ys).cpu().numpy()
        pred_ys = np.clip(pred_ys, SM_MIN, SM_MAX)

        pred_map = np.full((grid_info["H"], grid_info["W"]), np.nan, dtype=np.float32)
        pred_map[dataset.rows, dataset.cols] = pred_ys
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


def build_model() -> PixelNoisePredictor:
    return PixelNoisePredictor.from_pretrained(DDPM_PIXEL_MODEL_PATH)


@torch.no_grad()
def inference(model: PixelNoisePredictor, dataset: DDPMPixelInferenceDataset, device: str) -> torch.Tensor:
    model = model.to(device)
    scheduler = build_scheduler()
    model.eval()

    pred_ys_list = []
    data_loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=False)  # type: ignore[arg-type]
    for batch_xs, batch_pos, batch_valid_mask, batch_dates, batch_image_baseline in tqdm(data_loader):
        batch_xs = batch_xs.to(device)
        batch_pos = batch_pos.to(device)
        batch_valid_mask = batch_valid_mask.to(device)
        batch_image_baseline = batch_image_baseline.to(device).unsqueeze(1)
        inference_seed = build_inference_seed(batch_dates[0])

        batch_pred_residuals = reverse_diffuse(
            model, scheduler, batch_xs, batch_pos, batch_valid_mask, batch_dates,
            INFERENCE_STEP_NUM, device=device, seed=inference_seed
        )
        batch_pred_ys = batch_pred_residuals + batch_image_baseline
        batch_pred_ys = batch_pred_ys.reshape(-1).cpu()
        pred_ys_list.append(batch_pred_ys)

    return torch.cat(pred_ys_list, dim=0)


@torch.no_grad()
def reverse_diffuse(model: PixelNoisePredictor, scheduler: DDPMScheduler,
                    xs: torch.Tensor, pos: torch.Tensor, valid_mask: torch.Tensor, dates: List[str],
                    inference_step_num: int, device: str, seed: int) -> torch.Tensor:
    model.eval()
    batch_size = xs.shape[0]
    generator = torch.Generator().manual_seed(seed)
    ys = torch.randn(batch_size, 1, dtype=xs.dtype, generator=generator, device="cpu").to(device)
    scheduler.set_timesteps(inference_step_num)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((batch_size,), timestep.item(), device=device, dtype=torch.long)
        pred_x0 = model.forward(
            ys, xs, timesteps,
            pos=pos, dates=dates,
            valid_mask=valid_mask,
        )
        step_out = scheduler.step(model_output=pred_x0, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


def build_inference_seed(date_str: str) -> int:
    date_value = int(datetime.strptime(date_str, "%Y%m%d").strftime("%Y%m%d"))
    return (BASE_INFERENCE_SEED + date_value) % (2 ** 31 - 1)


if __name__ == "__main__":
    main()
