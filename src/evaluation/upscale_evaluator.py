#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of Upscaled Results
  @Author Chris
  @Date 2026/1/20
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from constants import *
from datasets.dataset import DataStore, GridInfoStore
from evaluation.evaluator import Evaluator
from utils.date_util import get_valid_dates

METHOD_NAME = RF_NAME

if METHOD_NAME == RF_NAME:
    SRC_DIR_PATH = RF_DIR_PATH
elif METHOD_NAME == DDPM_NAME:
    SRC_DIR_PATH = INFERENCE_DIR_PATH


def main():
    data_store = DataStore(resolution=RESOLUTION_36KM)
    grid_info = GridInfoStore(resolution=RESOLUTION_36KM).get()

    H, W = grid_info["H"], grid_info["W"]
    dates = get_valid_dates()
    all_pred, all_true, all_masks, all_dates = [], [], [], []
    for date in dates:
        src_path = os.path.join(SRC_DIR_PATH, RESOLUTION_1KM, f"{date}{TIFF_SUFFIX}")
        pred_y = upscale_to_36km(src_path, grid_info)
        true_y = data_store.get(SM_NAME, date).astype(np.float32)

        valid = (~np.isnan(pred_y)) & (~np.isnan(true_y))
        all_pred.append(pred_y.reshape(-1))
        all_true.append(true_y.reshape(-1))
        all_masks.append(valid.reshape(-1).astype(np.float32))
        all_dates.extend([date] * (H * W))

    pred = np.concatenate(all_pred).astype(np.float32)
    true = np.concatenate(all_true).astype(np.float32)
    masks = np.concatenate(all_masks).astype(np.float32)
    rows = np.tile(grid_info["rows"].reshape(-1), len(dates)).astype(np.int32)
    cols = np.tile(grid_info["cols"].reshape(-1), len(dates)).astype(np.int32)

    # Evaluate
    evaluator = Evaluator()

    evaluator.print_overall(pred, true, masks,
                            title=f"Overall Evaluation: Upscale {METHOD_NAME}")

    df_date = evaluator.evaluate_by_date(pred, true, masks, all_dates)
    out_date_csv = os.path.join(RESULT_DIR_PATH, f"Evaluation_Upscale_{METHOD_NAME}_By_Date.csv")
    df_date.to_csv(out_date_csv, index=False)

    df_site = evaluator.evaluate_by_site(pred, true, masks, all_dates, rows, cols)
    out_site_csv = os.path.join(RESULT_DIR_PATH, f"Evaluation_Upscale_{METHOD_NAME}_By_Site.csv")
    df_site.to_csv(out_site_csv, index=False)

    evaluator.evaluate_by_spatial_distribution(df_site, height=H, width=W)


def upscale_to_36km(src_path: str, dst_grid_info: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    dst_h, dst_w = dst_grid_info["H"], dst_grid_info["W"]
    dst_transform = dst_grid_info["transform"]
    dst_crs = dst_grid_info["crs"]

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


if __name__ == "__main__":
    main()
