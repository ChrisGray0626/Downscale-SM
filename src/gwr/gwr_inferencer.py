#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Geographically Weighted Regression Inferencer
@Author Chris
@Date 2026/1/27
"""
import pickle

import numpy as np
from mgwr.gwr import Gaussian, _compute_betas_gwr
from tqdm import tqdm

from constants import *
from datasets.common_data_store import GridInfoStore
from gwr.gwr_dataset import GWRInferenceDataset
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

RESOLUTION = RESOLUTION_1KM


def main():
    with open(GWR_MODEL_PATH, "rb") as f:
        model, _, _ = pickle.load(f)

    grid_info = GridInfoStore(RESOLUTION).get()
    H, W = grid_info["H"], grid_info["W"]
    dst_dir_path = os.path.join(GWR_DIR_PATH, RESOLUTION)
    os.makedirs(dst_dir_path, exist_ok=True)

    for date in tqdm(get_valid_dates(), desc=f"Inference {RESOLUTION}"):
        inf_dataset = GWRInferenceDataset(date=date, resolution=RESOLUTION)
        lons, lats, X_pred, rows, cols = inf_dataset.get_all()
        X_pred = X_pred.astype(np.float64)
        pos_pred = np.column_stack([lons, lats])
        pred_norm = predict_gwr(model, pos_pred, X_pred)
        pred_ys = inf_dataset.denorm_y(pred_norm).astype(np.float32)
        pred_map = np.full((H, W), np.nan, dtype=np.float32)
        pred_map[rows, cols] = pred_ys

        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(
            pred_map,
            dst_file_path,
            transform=grid_info["transform"],
            crs=grid_info["crs"],
        )


def predict_gwr(model, pos_pred: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    if not isinstance(model.family, Gaussian):
        raise NotImplementedError("Current GWR inference workaround only supports Gaussian family.")

    model.points = pos_pred
    if model.constant:
        p_pred = np.hstack([np.ones((len(x_pred), 1), dtype=np.float64), x_pred])
    else:
        p_pred = x_pred

    pred = np.empty(len(pos_pred), dtype=np.float64)
    for i in range(len(pos_pred)):
        wi = model._build_wi(i, model.bw).reshape(-1, 1)
        betas, _ = _compute_betas_gwr(model.y, model.X, wi)
        pred[i] = float(np.dot(p_pred[i], betas).reshape(-1)[0])
    model.points = None
    return pred.astype(np.float32)


if __name__ == "__main__":
    main()
