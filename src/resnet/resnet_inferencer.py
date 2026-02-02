#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet inference: load model, predict per date, write TIFF.
@Author Chris
@Date 2026
"""

import torch
from tqdm import tqdm

from constants import *
from datasets.dataset import GridInfoStore
from model.module import build_device
from resnet.resnet_dataset import ResNetInferenceDataset
from resnet.resnet_model import load_checkpoint
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

RESOLUTION = RESOLUTION_36KM


def main():
    device = build_device()
    model, y_mean, y_std = load_checkpoint(RESNET_MODEL_PATH, device)
    model.eval()

    grid_info = GridInfoStore(RESOLUTION).get()
    dst_dir = os.path.join(RESNET_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir, exist_ok=True)

    for date in tqdm(get_valid_dates(), desc="Inference"):
        dataset = ResNetInferenceDataset(date, RESOLUTION)
        xs = dataset.get_all()
        xs = xs.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_y = model(xs).squeeze(0).squeeze(0).cpu().numpy()
        pred_y = pred_y * y_std + y_mean
        write_tiff(pred_y, os.path.join(dst_dir, f"{date}{TIFF_SUFFIX}"), transform=grid_info["transform"],
                   crs=grid_info["crs"])


if __name__ == "__main__":
    main()
