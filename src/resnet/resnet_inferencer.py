#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet inference: load model, predict per date, write TIFF.
@Author Chris
@Date 2026
"""

import numpy as np
import torch
from tqdm import tqdm

from constants import *
from datasets.dataset import GridInfoStore
from model.module import build_device
from resnet_dataset import ResNetInferenceDataset
from resnet_model import load_checkpoint
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

RESOLUTION = RESOLUTION_1KM


def main():
    device = build_device()
    model, y_mean, y_std = load_checkpoint(RESNET_MODEL_PATH, device)
    model.eval()

    grid_info = GridInfoStore(RESOLUTION).get()
    dst_dir = os.path.join(RESNET_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir, exist_ok=True)

    for date in tqdm(get_valid_dates(), desc="Inference"):
        inf_dataset = ResNetInferenceDataset(date, RESOLUTION)
        data = inf_dataset.get_all()
        x = data["x"].unsqueeze(0).to(device)
        valid = data["valid"]
        with torch.no_grad():
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
        pred_denorm = pred * y_std + y_mean
        pred_map = np.where(valid, pred_denorm, np.nan).astype(np.float32)
        write_tiff(pred_map, os.path.join(dst_dir, f"{date}{TIFF_SUFFIX}"), transform=grid_info["transform"],
                   crs=grid_info["crs"])


if __name__ == "__main__":
    main()
