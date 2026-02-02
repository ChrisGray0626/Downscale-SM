#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Geographically Weighted Regression Dataset - inherits common, get_all exposes only (pos, y, X) / (pos, xs, rows, cols).
@Author Chris
@Date 2026/1/27
"""

from constants import GWR_DIR_PATH, LONGITUDE_NAME, LATITUDE_NAME, X_NAME, Y_NAME, ROW_NAME, COL_NAME
from datasets.common_dataset import CommonTrainDataset, CommonInferenceDataset
from utils.data_store import TiffStore


class GWRTrainDataset(CommonTrainDataset):

    def __init__(self):
        super().__init__(flat=True, filter_valid=True)

    def get_all(self):
        data = super().get_all()
        return data[LONGITUDE_NAME], data[LATITUDE_NAME], data[X_NAME], data[Y_NAME].reshape(-1, 1)


class GWRInferenceDataset(CommonInferenceDataset):

    def __init__(self, date: str, resolution: str):
        super().__init__(date, resolution, flat=True, filter_valid=True)

    def get_all(self):
        data = super().get_all()
        return data[LONGITUDE_NAME], data[LATITUDE_NAME], data[X_NAME], data[ROW_NAME], data[COL_NAME]


class GWRResultStore(TiffStore):
    def __init__(self, resolution: str):
        base_dir = GWR_DIR_PATH
        super().__init__(base_dir, resolution)
