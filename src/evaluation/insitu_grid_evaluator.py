#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of In-situ vs Product
  @Author Chris
  @Date 2026/1/27
"""

import numpy as np

from constants import *
from datasets import data_store_factory
from datasets.common_data_store import GridInfoStore, InsituStore
from evaluation.evaluator import Evaluator

PROD_NAMES = [
    DDPM_IMAGE_NAME,
    DDPM_PIXEL_NAME,
    RF_NAME,
    RESNET_NAME,
    ESA_CCI_NAME,
    # SM_NAME,
    # GWR_NAME,
]
RESOLUTIONS = [
    RESOLUTION_36KM,
    RESOLUTION_1KM,
]


def main():
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    for resolution in RESOLUTIONS:
        for product_name in PROD_NAMES:
            dataset = InsituGridEvalDataset(product_name, resolution=resolution)
            pred_map, insitu_map, insitu_masks, dates, rows, cols = dataset.get_all()
            grid_info = dataset.grid_info

            title = f"Overall Evaluation: {resolution} {product_name} vs InSitu Data"
            evaluator.print_overall(pred_map, insitu_map, insitu_masks, title=title)

            df_date = evaluator.evaluate_by_date(pred_map, insitu_map, insitu_masks, dates)

            dst_file_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_By_Date_{resolution}",
                                         f"{product_name}.csv")
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            df_date.to_csv(dst_file_path, index=False)

            df_site = evaluator.evaluate_by_site(pred_map, insitu_map, insitu_masks, dates, rows, cols)

            dst_file_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_By_Site_{resolution}",
                                         f"{product_name}.csv")
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            df_date.to_csv(dst_file_path, index=False)

            evaluator.evaluate_by_spatial_distribution(
                df_site, height=grid_info["H"], width=grid_info["W"],
                title=f"{resolution} {product_name} vs InSitu Data",
            )


class InsituGridEvalDataset:
    def __init__(self, product_name, resolution):
        self.product_name = product_name
        self.resolution = resolution
        self.pred_store = data_store_factory.build(product_name, resolution, is_correction=True)
        self.insitu_store = InsituStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._grid_info = self.grid_info_store.get()

    @property
    def _dates(self):
        return sorted(set(self.insitu_store.list_date()) & set(self.pred_store.list_date()))

    def get_all(self):
        H, W = self._grid_info["H"], self._grid_info["W"]
        rows = self._grid_info["rows"]
        cols = self._grid_info["cols"]

        all_pred, all_insitu, all_masks, all_dates, all_rows, all_cols = [], [], [], [], [], []

        for date in self._dates:
            pred_map = self.pred_store.get(date).astype(np.float32)
            insitu_map = self.insitu_store.get(date).astype(np.float32)

            pred_mask = (~np.isnan(pred_map)).astype(np.float32)
            insitu_mask = (~np.isnan(insitu_map)).astype(np.float32)
            valid = (pred_mask > 0) & (insitu_mask > 0)

            all_pred.append(pred_map.reshape(-1))
            all_insitu.append(insitu_map.reshape(-1))
            all_masks.append(valid.reshape(-1).astype(np.float32))
            all_dates.extend([date] * (H * W))
            all_rows.append(rows.reshape(-1))
            all_cols.append(cols.reshape(-1))

        pred_map = np.concatenate(all_pred).astype(np.float32)
        insitu_map = np.concatenate(all_insitu).astype(np.float32)
        valid_masks = np.concatenate(all_masks).astype(np.float32)
        rows_flat = np.concatenate(all_rows).astype(np.int32)
        cols_flat = np.concatenate(all_cols).astype(np.int32)

        return pred_map, insitu_map, valid_masks, all_dates, rows_flat, cols_flat

    @property
    def grid_info(self):
        return self._grid_info


if __name__ == "__main__":
    main()
