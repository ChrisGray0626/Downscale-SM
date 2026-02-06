#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Check data coverage between training data and insitu data by date
@Author Chris
@Date 2025/12/12
"""
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from constants import RESULT_DIR_PATH, RESOLUTION_36KM, NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME, \
    SM_NAME, IN_SITU_NAME
from datasets.common_data_store import GridInfoStore, ModelDataStore
from utils.date_util import get_valid_dates


def main():
    dataset = DataCoverageDataset(resolution=RESOLUTION_36KM)
    rows, cols, feature_values, train_valid, insitu_valid, dates = dataset.get_all()

    df = pd.DataFrame({
        'Date': dates,
        'Row': rows,
        'Col': cols,
        NDVI_NAME: feature_values[NDVI_NAME],
        LST_NAME: feature_values[LST_NAME],
        ALBEDO_NAME: feature_values[ALBEDO_NAME],
        PRECIPITATION_NAME: feature_values[PRECIPITATION_NAME],
        DEM_NAME: feature_values[DEM_NAME],
        SM_NAME: feature_values[SM_NAME],
        IN_SITU_NAME: feature_values[IN_SITU_NAME],
        'Train_Valid': train_valid,
        'InSitu_Valid': insitu_valid
    })

    os.makedirs(RESULT_DIR_PATH, exist_ok=True)
    df.to_csv(os.path.join(RESULT_DIR_PATH, 'Coverage_Data.csv'), index=False)


class DataCoverageDataset(Dataset):

    def __init__(self, resolution: str):
        self.resolution = resolution
        self.data_store = ModelDataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)

        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.rows = grid_info["rows"]
        self.cols = grid_info["cols"]
        self.feature_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME, SM_NAME, IN_SITU_NAME]

    def get(self, date: str) -> tuple:
        features = {}
        for name in self.feature_names:
            try:
                if name == DEM_NAME:
                    features[name] = self.data_store.get(name, None)
                else:
                    features[name] = self.data_store.get(name, date)
            except FileNotFoundError:
                features[name] = np.full((self.H, self.W), np.nan, dtype=np.float32)

        train_valid = ~np.isnan(features[NDVI_NAME]) & ~np.isnan(features[LST_NAME]) & \
                      ~np.isnan(features[ALBEDO_NAME]) & ~np.isnan(features[PRECIPITATION_NAME]) & \
                      ~np.isnan(features[DEM_NAME]) & ~np.isnan(features[SM_NAME])
        insitu_valid = ~np.isnan(features[IN_SITU_NAME])

        all_positions_mask = train_valid | insitu_valid

        rows = self.rows[all_positions_mask]
        cols = self.cols[all_positions_mask]
        feature_values = {name: features[name][all_positions_mask] for name in self.feature_names}
        train_valid_mask = train_valid[all_positions_mask]
        insitu_valid_mask = insitu_valid[all_positions_mask]

        return rows, cols, feature_values, train_valid_mask, insitu_valid_mask

    def get_all(self) -> tuple:
        dates = get_valid_dates()
        all_rows, all_cols, all_dates = [], [], []
        all_feature_values = {name: [] for name in self.feature_names}
        all_train_valid, all_insitu_valid = [], []

        for date in dates:
            rows, cols, feature_values, train_valid, insitu_valid = self.get(date)

            all_rows.append(rows)
            all_cols.append(cols)
            all_dates.extend([date] * len(rows))
            for name in self.feature_names:
                all_feature_values[name].append(feature_values[name])
            all_train_valid.append(train_valid)
            all_insitu_valid.append(insitu_valid)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        feature_values = {name: np.concatenate(all_feature_values[name]) for name in self.feature_names}
        train_valid = np.concatenate(all_train_valid)
        insitu_valid = np.concatenate(all_insitu_valid)

        return rows, cols, feature_values, train_valid, insitu_valid, all_dates

if __name__ == "__main__":
    main()
