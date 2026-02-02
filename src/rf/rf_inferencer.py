#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Random Forest Inferencer
@Author Chris
@Date 2025/12/12
"""
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from constants import *
from datasets.dataset import GridInfoStore
from rf.rf_dataset import RFInferenceDataset
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

RESOLUTION = RESOLUTION_36KM


def main():
    model = build_model()

    grid_info = GridInfoStore(RESOLUTION).get()
    H, W = grid_info["H"], grid_info["W"]

    # Inference
    dst_dir_path = os.path.join(RF_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)
    for date in tqdm(get_valid_dates(), desc="Inference"):
        dataset = RFInferenceDataset(date=date, resolution=RESOLUTION)
        rows, cols, xs = dataset.get_all()
        pred_ys = model.predict(xs)
        pred_ys = dataset.denorm_y(pred_ys).astype(np.float32)

        pred_map = np.full((H, W), np.nan, dtype=np.float32)
        pred_map[rows, cols] = pred_ys

        # Save Inference Result
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(
            pred_map,
            dst_file_path,
            transform=grid_info["transform"],
            crs=grid_info["crs"]
        )


def build_model():
    with open(RF_MODEL_PATH, 'rb') as f:
        rf: RandomForestRegressor = pickle.load(f)

    return rf


if __name__ == "__main__":
    main()
