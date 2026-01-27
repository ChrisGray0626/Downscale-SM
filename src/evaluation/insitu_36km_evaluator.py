#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description 36km dataset vs in-situ evaluation
  @Author Chris
  @Date 2026/1/27
"""

import numpy as np

from constants import *
from datasets.dataset import GridInfoStore, InsituStore, CorrectionResultStore
from esa_cci.esa_cci_dataset import ESACCIStore
from evaluation.evaluator import Evaluator
from rf.rf_dataset import RFResultStore

PRODUCT_NAME = DDPM_NAME


def main():
    dataset = Insitu36kmEvalDataset(PRODUCT_NAME)
    pred_map, insitu_map, insitu_masks, dates, rows, cols = dataset.get_all()
    grid_info = dataset.grid_info

    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    evaluator.print_overall(
        pred_map, insitu_map, insitu_masks,
        title=f"Overall Evaluation: 36km {PRODUCT_NAME} vs InSitu Data",
    )

    print("\n" + "=" * 60)
    print(f"Evaluation by Date:  36km {PRODUCT_NAME} vs InSitu Data")
    print("=" * 60)
    df_date = evaluator.evaluate_by_date(pred_map, insitu_map, insitu_masks, dates)
    df_date.to_csv(os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_36km_{PRODUCT_NAME}_By_Date.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"Evaluation by Site:  36km {PRODUCT_NAME} vs InSitu Data")
    print("=" * 60)
    df_site = evaluator.evaluate_by_site(pred_map, insitu_map, insitu_masks, dates, rows, cols)
    df_site.to_csv(os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_36km_{PRODUCT_NAME}_By_Site.csv"), index=False)

    evaluator.evaluate_by_spatial_distribution(df_site, height=grid_info["H"], width=grid_info["W"])


class Insitu36kmEvalDataset:
    def __init__(self, product_name):
        self.product_name = product_name
        self.resolution = RESOLUTION_36KM
        self.pred_store = self._build_pred_store(product_name)
        self.insitu_store = InsituStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._grid_info = self.grid_info_store.get()

    @staticmethod
    def _build_pred_store(product_name: str):
        if product_name == ESA_CCI_NAME:
            return ESACCIStore(resolution=RESOLUTION_36KM)
        elif product_name == RF_NAME:
            return RFResultStore(resolution=RESOLUTION_36KM)
        elif product_name == DDPM_NAME:
            return CorrectionResultStore(resolution=RESOLUTION_36KM)
        raise ValueError(f"Unknown product name: {product_name}")

    def get_all(self):
        H, W = self._grid_info["H"], self._grid_info["W"]
        rows = self._grid_info["rows"]
        cols = self._grid_info["cols"]

        all_pred, all_insitu, all_masks, all_dates, all_rows, all_cols = [], [], [], [], [], []

        for date in self.pred_store.list_date():
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
