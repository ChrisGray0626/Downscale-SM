#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of Product vs SMAP (36km)
  @Author Chris
  @Date 2026/2/2
"""

import numpy as np

from constants import *
from datasets import data_store_factory
from datasets.common_data_store import GridInfoStore, ModelDataStore
from evaluation.evaluator import Evaluator
from utils.date_util import get_valid_dates

PROD_NAMES = [
    DDPM_IMAGE_NAME,
    DDPM_PIXEL_NAME,
    RF_NAME,
    RESNET_NAME,
    # GWR_NAME,
    IN_SITU_NAME,
]
RESOLUTION = RESOLUTION_36KM


def main():
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    for product_name in PROD_NAMES:
        dataset = SmapGridEvalDataset(product_name, resolution=RESOLUTION)
        pred_map, sm_map, valid_masks, all_dates, rows_flat, cols_flat = dataset.get_all()
        grid_info = dataset.grid_info

        title = f"Overall Evaluation: {RESOLUTION} {product_name} vs SM Data"
        evaluator.print_overall(pred_map, sm_map, valid_masks, title=title)

        df_date = evaluator.evaluate_by_date(pred_map, sm_map, valid_masks, all_dates)
        dst_file_path = os.path.join(
            RESULT_DIR_PATH, f"Evaluation_SMAP_By_Date_{RESOLUTION}", f"{product_name}.csv"
        )
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        df_date.to_csv(dst_file_path, index=False)

        df_site = evaluator.evaluate_by_site(
            pred_map, sm_map, valid_masks, all_dates, rows_flat, cols_flat
        )
        dst_file_path = os.path.join(
            RESULT_DIR_PATH, f"Evaluation_SMAP_By_Site_{RESOLUTION}", f"{product_name}.csv"
        )
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        df_site.to_csv(dst_file_path, index=False)

        evaluator.evaluate_by_spatial_distribution(
            df_site, height=grid_info["H"], width=grid_info["W"],
            title=f"{RESOLUTION} {product_name} vs SMAP Data",
        )


class SmapGridEvalDataset:

    def __init__(self, product_name: str, resolution: str):
        self.product_name = product_name
        self.resolution = resolution
        self.pred_store = data_store_factory.build(product_name, resolution)
        self.data_store = ModelDataStore(resolution=resolution)
        self.grid_info_store = GridInfoStore(resolution=resolution)
        self._grid_info = self.grid_info_store.get()

    @property
    def _dates(self):
        pred_dates = set(self.pred_store.list_date())
        sm_dates = set(get_valid_dates())
        return sorted(pred_dates & sm_dates)

    def get_all(self):
        H, W = self._grid_info["H"], self._grid_info["W"]
        rows = self._grid_info["rows"]
        cols = self._grid_info["cols"]

        all_pred, all_sm, all_masks, all_dates, all_rows, all_cols = [], [], [], [], [], []

        for date in self._dates:
            pred_map = self.pred_store.get(date).astype(np.float32)
            sm_map = self.data_store.get(SM_NAME, date).astype(np.float32)

            pred_mask = (~np.isnan(pred_map)).astype(np.float32)
            sm_mask = (~np.isnan(sm_map)).astype(np.float32)
            valid = (pred_mask > 0) & (sm_mask > 0)

            all_pred.append(pred_map.reshape(-1))
            all_sm.append(sm_map.reshape(-1))
            all_masks.append(valid.reshape(-1).astype(np.float32))
            all_dates.extend([date] * (H * W))
            all_rows.append(rows.reshape(-1))
            all_cols.append(cols.reshape(-1))

        pred_map = np.concatenate(all_pred).astype(np.float32)
        sm_map = np.concatenate(all_sm).astype(np.float32)
        valid_masks = np.concatenate(all_masks).astype(np.float32)
        rows_flat = np.concatenate(all_rows).astype(np.int32)
        cols_flat = np.concatenate(all_cols).astype(np.int32)

        return pred_map, sm_map, valid_masks, all_dates, rows_flat, cols_flat

    @property
    def grid_info(self):
        return self._grid_info


if __name__ == "__main__":
    main()
