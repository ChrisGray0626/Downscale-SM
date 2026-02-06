#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description DDPM_Image-based Soil Moisture Downscaling Corrector
@Author Chris
@Date 2025/12/12
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import *
from datasets.common_data_store import GridInfoStore
from utils.date_util import get_valid_dates
from utils.raster_util import write_tiff

BATCH_SIZE = 16384
SM_MIN = 0.02
SM_MAX = 0.5

RESOLUTION = RESOLUTION_1KM


# TODO
def main():
    grid_info = GridInfoStore(RESOLUTION).get()
    correction_dir_path = os.path.join(DDPM_IMAGE_CORRECTION_DIR_PATH, RESOLUTION)
    os.makedirs(correction_dir_path, exist_ok=True)

    # Collect Data for Bias Correction Training
    all_pred_ys_insitu = []
    all_insitus = []
    all_aux_feats_insitu = []

    dates = get_valid_dates()
    for date in tqdm(dates, desc="Collecting"):
        correction_dataset = CorrectionDataset(date=date, resolution=RESOLUTION)

        pred_ys = correction_dataset.pred_map[correction_dataset.rows, correction_dataset.cols]

        insitu_mask = ~np.isnan(correction_dataset.insitu)
        if insitu_mask.sum() > 0:
            all_pred_ys_insitu.append(pred_ys[insitu_mask])
            all_insitus.append(correction_dataset.insitu[insitu_mask])
            all_aux_feats_insitu.append(correction_dataset.xs[insitu_mask])

    # Train Bias Corrector
    all_pred_ys_insitu_concat = np.concatenate(all_pred_ys_insitu)
    all_insitus_concat = np.concatenate(all_insitus)
    all_aux_feats_insitu_concat = np.concatenate(all_aux_feats_insitu)

    rf_corrector = BiasCorrector()
    rf_corrector.train(
        pred_ys=all_pred_ys_insitu_concat,
        insitus=all_insitus_concat,
        aux_feats=all_aux_feats_insitu_concat,
        verbose=True
    )

    # Correction
    for date in tqdm(dates, desc="Correction"):
        correction_dataset = CorrectionDataset(date=date, resolution=RESOLUTION)

        all_pred_ys_corrected = []
        data_loader = DataLoader(correction_dataset, batch_size=min(BATCH_SIZE, len(correction_dataset)), shuffle=False)
        for batch_xs, batch_pred_ys, _, _ in data_loader:
            batch_xs = batch_xs.numpy().astype(np.float32)
            batch_pred_ys = batch_pred_ys.numpy().astype(np.float32)

            batch_pred_ys_corrected = rf_corrector.predict(
                pred_ys=batch_pred_ys,
                aux_feats=batch_xs
            )

            all_pred_ys_corrected.append(batch_pred_ys_corrected)

        pred_ys_corrected = np.concatenate(all_pred_ys_corrected)

        # Clip
        pred_ys_corrected = np.clip(pred_ys_corrected, SM_MIN, SM_MAX)

        # Save Corrected Result
        pred_map = np.full((grid_info["H"], grid_info["W"]), np.nan, dtype=np.float32)
        pred_map[correction_dataset.rows, correction_dataset.cols] = pred_ys_corrected
        dst_file_path = os.path.join(correction_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


class CorrectionDataset(Dataset):

    def __init__(self, date: str, resolution: str):
        self.date = date
        self.resolution = resolution
        self.inference_result_store = DDPMImageInferenceResultStore(resolution=self.resolution)
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.train_dataset = TrainDataset()

        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        pred_map = self.inference_result_store.get(self.date)

        xs = np.stack([
            self.data_store.get(name, self.date) for name in
            [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
        ], axis=-1)

        insitu_map = self.data_store.get(IN_SITU_NAME, self.date)

        grid_info = self.grid_info_store.get()
        H, W = grid_info["H"], grid_info["W"]

        self.pred_map = pred_map.astype(np.float32)
        self.xs = xs.reshape(H * W, -1).astype(np.float32)
        self.insitu = insitu_map.flatten().astype(np.float32)
        self.grid_info = grid_info
        self.rows_full = grid_info["rows"].flatten()
        self.cols_full = grid_info["cols"].flatten()

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1)

        self.xs = self.xs[valid]
        self.rows = self.rows_full[valid]
        self.cols = self.cols_full[valid]
        self.insitu = self.insitu[valid]

    def _norm(self):
        x_mean = self.train_dataset.x_mean
        x_std = self.train_dataset.x_std.copy()
        x_std[x_std == 0] = 1.0
        self.xs = (self.xs - x_mean) / x_std

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).float()
        pred_y = torch.tensor(self.pred_map[self.rows[idx], self.cols[idx]], dtype=torch.float32)
        row = torch.tensor(self.rows[idx], dtype=torch.long)
        col = torch.tensor(self.cols[idx], dtype=torch.long)
        return xs, pred_y, row, col

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

            print(f"RF Bias Corrector Training:")
            print(f"  Training samples: {len(x_train):,}, Validation samples: {len(x_val):,}")
            print(f"  Validation RMSE: {rmse:.6f}, R: {r:.4f}")
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                print(f"  OOB Score: {self.model.oob_score_:.4f}")

        return self.model

    def predict(self, pred_ys: np.ndarray, aux_feats: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        return self.model.predict(X).astype(np.float32)


if __name__ == "__main__":
    main()
