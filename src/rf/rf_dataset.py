#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Random Forest Dataset - Date-based (not pixel-based)
@Author Chris
@Date 2025/12/12
"""

from constants import *
from datasets.common_dataset import CommonTrainDataset, CommonInferenceDataset
from utils.data_store import TiffStore


class RFTrainDataset(CommonTrainDataset):

    def get_all(self):
        data = super().get_all()
        return data[X_NAME], data[Y_NAME]


class RFInferenceDataset(CommonInferenceDataset):

    def get_all(self):
        data = super().get_all()
        return data[ROW_NAME], data[COL_NAME], data[X_NAME]


class RFResultStore(TiffStore):
    def __init__(self, resolution: str):
        base_dir = RF_DIR_PATH
        super().__init__(base_dir, resolution)
