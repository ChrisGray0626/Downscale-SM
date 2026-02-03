#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM-based Soil Moisture Downscaling Inferencer
@Author Chris
@Date 2025/12/12
"""

import numpy as np
import torch
from tqdm import tqdm

from constants import *
from datasets.dataset import GridInfoStore
from model.ddpm_dataset import DDPMInferenceDataset
from model.module import NoisePredictor, build_device
from model.trainer import build_scheduler, reverse_diffuse
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

INFERENCE_STEP_NUM = 50
SM_MIN = 0.02
SM_MAX = 0.5
RESOLUTION = RESOLUTION_36KM


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
            model, scheduler, xs, [date_str], INFERENCE_STEP_NUM, device, insitu_stats
        )
        pred_y = pred_y.squeeze(0).squeeze(0).cpu().numpy()
        pred_y = dataset.train_dataset.denorm_y(pred_y)
        pred_y = np.clip(pred_y, SM_MIN, SM_MAX)

        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_y, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


def build_model() -> NoisePredictor:
    return NoisePredictor.from_pretrained(DDPM_MODEL_PATH)


if __name__ == "__main__":
    main()
