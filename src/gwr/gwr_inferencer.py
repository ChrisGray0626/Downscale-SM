#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Geographically Weighted Regression Inferencer
@Author Chris
@Date 2026/1/27
"""
import pickle

import joblib
import numpy as np
from tqdm import tqdm

from constants import *
from datasets.common_data_store import GridInfoStore
from gwr.gwr_dataset import GWRInferenceDataset
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff
from utils.util import suppress_linalg

RESOLUTION = RESOLUTION_1KM


def main():
    with open(GWR_MODEL_PATH, "rb") as f:
        model, exog_scale, exog_resid = pickle.load(f)

    grid_info = GridInfoStore(RESOLUTION).get()
    H, W = grid_info["H"], grid_info["W"]
    dst_dir_path = os.path.join(GWR_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)

    for date in tqdm(get_valid_dates(), desc=f"Inference {RESOLUTION}"):
        inf_dataset = GWRInferenceDataset(date=date, resolution=RESOLUTION)
        lons, lats, X_pred, rows, cols = inf_dataset.get_all()
        X_pred = X_pred.astype(np.float64)
        pos_pred = np.column_stack([lons, lats])
        with joblib.parallel_backend("loky", initializer=suppress_linalg, initargs=()):  # type: ignore[call-arg]
            pred_results = model.predict(pos_pred, X_pred, exog_scale=exog_scale,
                                         exog_resid=exog_resid)  # type: ignore[call-arg]
        pred_ys = inf_dataset.denorm_y(pred_results.predy.flatten()).astype(np.float32)
        pred_map = np.full((H, W), np.nan, dtype=np.float32)
        pred_map[rows, cols] = pred_ys

        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(
            pred_map,
            dst_file_path,
            transform=grid_info["transform"],
            crs=grid_info["crs"],
        )


if __name__ == "__main__":
    main()
