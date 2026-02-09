#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Bias Correction for DDPM Inference Products using Random Forest
  @Author Chris
  @Date 2025/12/12
"""

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import *
from datasets import data_store_factory
from datasets.common_data_store import GridInfoStore, ModelDataStore
from utils.raster_util import write_tiff

PROD_NAME = DDPM_PIXEL_NAME
RESOLUTION = RESOLUTION_1KM


def main():
    dataset = CorrectionDataset(PROD_NAME, resolution=RESOLUTION)
    pred_ys, insitus, aux_feats = dataset.get_training_data()
    rf_corrector = BiasCorrector()
    rf_corrector.train(pred_ys=pred_ys, insitus=insitus, aux_feats=aux_feats, verbose=True)

    correction_store = data_store_factory.build(PROD_NAME, RESOLUTION, is_correction=True)
    dst_dir_path = os.path.join(correction_store.base_dir, correction_store.resolution)
    os.makedirs(dst_dir_path, exist_ok=True)
    grid_info = dataset.grid_info

    for date in tqdm(dataset.dates, desc=f"Correction {PROD_NAME} {RESOLUTION}"):
        xs, pred_ys, rows, cols = dataset.get_correction_data_for_date(date)
        pred_corrected = rf_corrector.predict(pred_ys=pred_ys, aux_feats=xs)
        pred_corrected = np.clip(pred_corrected, SM_MIN, SM_MAX)

        pred_map = np.full((grid_info["H"], grid_info["W"]), np.nan, dtype=np.float32)
        pred_map[rows, cols] = pred_corrected
        dst_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")  # type: ignore
        write_tiff(pred_map, dst_path, transform=grid_info["transform"], crs=grid_info["crs"])  # type: ignore


class CorrectionDataset:

    def __init__(self, product_name: str, resolution: str):
        self.product_name = product_name
        self.resolution = resolution
        self.inference_store = data_store_factory.build(product_name, resolution, is_correction=False)
        self.data_store = ModelDataStore(resolution=resolution)
        self.grid_info_store = GridInfoStore(resolution=resolution)
        self._grid_info = self.grid_info_store.get()

        self._load_data()
        self._norm()

    @property
    def dates(self):
        return sorted(set(self.inference_store.list_date()))

    @property
    def grid_info(self):
        return self._grid_info

    def _load_data(self):
        dates = self.dates
        grid_info = self._grid_info
        H, W = grid_info["H"], grid_info["W"]
        rows_1d = grid_info["rows"].flatten()
        cols_1d = grid_info["cols"].flatten()
        n_pix = H * W

        pred_list, xs_list, insitu_list, rows_list, cols_list, dates_list = [], [], [], [], [], []

        for date in dates:
            pred_map = self.inference_store.get(date).astype(np.float32)
            xs = np.stack([self.data_store.get(name, date) for name in AUX_FEAT_NAMES], axis=-1)
            insitu = self.data_store.get(IN_SITU_NAME, date).flatten().astype(np.float32)
            pred_list.append(pred_map.reshape(-1))
            xs_list.append(xs.reshape(n_pix, -1).astype(np.float32))
            insitu_list.append(insitu)
            rows_list.append(rows_1d)
            cols_list.append(cols_1d)
            dates_list.append(np.full(n_pix, date, dtype=object))

        self.pred_ys = np.concatenate(pred_list)
        self.xs = np.concatenate(xs_list, axis=0)
        self.insitu = np.concatenate(insitu_list)
        self.rows = np.concatenate(rows_list)
        self.cols = np.concatenate(cols_list)
        self.dates_flat = np.concatenate(dates_list)

    def _norm(self):
        self.x_mean = np.nanmean(self.xs, axis=0).astype(np.float32)
        self.x_std = np.nanstd(self.xs, axis=0).astype(np.float32)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std
        self.xs = np.nan_to_num(self.xs, nan=0.0, posinf=0.0, neginf=0.0)

    def get_training_data(self) -> Tuple:
        mask = ~np.isnan(self.insitu)
        return (
            self.pred_ys[mask].astype(np.float32),
            self.insitu[mask].astype(np.float32),
            self.xs[mask].astype(np.float32),
        )

    def get_correction_data_for_date(self, date: str) -> Tuple:
        mask = (self.dates_flat == date)
        return (
            self.xs[mask],
            self.pred_ys[mask],
            self.rows[mask],
            self.cols[mask],
        )


class BiasCorrector:

    def __init__(self, n_estimators: int = 300, max_depth: int = None,
                 max_features: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.model = None

    def train(self, pred_ys: np.ndarray, insitus: np.ndarray,
              aux_feats: np.ndarray, verbose: bool = True):
        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        y = insitus.astype(np.float32)

        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        n_samples = len(X)
        n_est = min(self.n_estimators, max(50, n_samples // 10))

        self.model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=-1,
            random_state=self.random_state,
            oob_score=n_samples > 50
        )
        self.model.fit(X, y)

        if verbose:
            y_pred_val = self.model.predict(x_val)
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r = np.nan
            if np.std(y_val) > 1e-12 and np.std(y_pred_val) > 1e-12:
                r = float(np.corrcoef(y_val, y_pred_val)[0, 1])

            print("RF Bias Corrector Training:")
            print(f"  Training samples: {len(x_train):,}, Validation samples: {len(x_val):,}")
            print(f"  Validation RMSE: {rmse:.6f}, R: {r:.4f}")

        return self.model

    def predict(self, pred_ys: np.ndarray, aux_feats: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        return self.model.predict(X).astype(np.float32)


if __name__ == "__main__":
    main()
