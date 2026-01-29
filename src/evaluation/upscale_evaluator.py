#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of Upscaled Results (1km -> 36km)
  @Author Chris
  @Date 2026/1/20
"""

import numpy as np
from rasterio.warp import reproject, Resampling

from constants import *
from datasets.dataset import ModelDataStore, GridInfoStore
from evaluation.evaluator import Evaluator
from evaluation.pred_store import build_pred_store
from utils.date_util import get_valid_dates

METHOD_NAMES = [RF_NAME, DDPM_NAME]


def main():
    data_store = ModelDataStore(resolution=RESOLUTION_36KM)
    grid_info_36km = GridInfoStore(resolution=RESOLUTION_36KM).get()
    grid_info_1km = GridInfoStore(resolution=RESOLUTION_1KM).get()

    H, W = grid_info_36km["H"], grid_info_36km["W"]
    dates = get_valid_dates()
    evaluator = Evaluator()

    for method_name in METHOD_NAMES:
        pred_store_1km = build_pred_store(method_name, resolution=RESOLUTION_1KM)

        all_pred, all_true, all_masks, all_dates = [], [], [], []
        for date in dates:
            pred_1km = pred_store_1km.get(date).astype(np.float32)
            pred_y = upscale_to_36km(pred_1km, grid_info_1km, grid_info_36km)
            true_y = data_store.get(SM_NAME, date).astype(np.float32)

            valid = (~np.isnan(pred_y)) & (~np.isnan(true_y))
            all_pred.append(pred_y.reshape(-1))
            all_true.append(true_y.reshape(-1))
            all_masks.append(valid.reshape(-1).astype(np.float32))
            all_dates.extend([date] * (H * W))

        pred = np.concatenate(all_pred).astype(np.float32)
        true = np.concatenate(all_true).astype(np.float32)
        masks = np.concatenate(all_masks).astype(np.float32)
        rows = np.tile(grid_info_36km["rows"].reshape(-1), len(dates)).astype(np.int32)
        cols = np.tile(grid_info_36km["cols"].reshape(-1), len(dates)).astype(np.int32)

        title = f"Overall Evaluation: Upscale {method_name} (1km -> 36km)"
        evaluator.print_overall(pred, true, masks, title=title)

        df_date = evaluator.evaluate_by_date(pred, true, masks, all_dates)
        out_date_csv = os.path.join(RESULT_DIR_PATH, f"Evaluation_Upscale_{method_name}_By_Date.csv")
        df_date.to_csv(out_date_csv, index=False)

        df_site = evaluator.evaluate_by_site(pred, true, masks, all_dates, rows, cols)
        out_site_csv = os.path.join(RESULT_DIR_PATH, f"Evaluation_Upscale_{method_name}_By_Site.csv")
        df_site.to_csv(out_site_csv, index=False)

        evaluator.evaluate_by_spatial_distribution(df_site, height=H, width=W)


def upscale_to_36km(src_data: np.ndarray, src_grid_info: dict, dst_grid_info: dict) -> np.ndarray:
    src_transform = src_grid_info["transform"]
    src_crs = src_grid_info["crs"]

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
