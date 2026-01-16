#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Check data coverage between training data and insitu data by date
@Author Chris
@Date 2025/12/12
"""
import os

import pandas as pd

from constants import RESULT_DIR_PATH, RESOLUTION_36KM, NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME, SM_NAME, IN_SITU_NAME
from dataset.dataset import DataCoverageDataset


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


if __name__ == "__main__":
    main()
