#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/7/29
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Constant import *
from util.tiff_util import read_tiff, read_tiff_data
from util.util import get_tgt_dates

IN_SITU_TIFF_DIR_PATH = os.path.join(RESULT_PATH, IN_SITU_NAME, CONVERTED_DIR_NAME)


def statistic(pre_result_data, in_situ_data):
    valid_mask = ((~np.isnan(pre_result_data)) & (~np.isnan(in_situ_data)))
    pre_result_data = pre_result_data[valid_mask]
    in_situ_data = in_situ_data[valid_mask]
    print(f"shape of pre_result_data: {pre_result_data.shape}")
    rmse = np.sqrt(mean_squared_error(in_situ_data, pre_result_data))
    mae = mean_absolute_error(in_situ_data, pre_result_data)
    bias = np.mean(pre_result_data - in_situ_data)
    r2 = r2_score(in_situ_data, pre_result_data)

    print("===== 精度评估结果 =====")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"Bias : {bias:.4f}")
    print(f"R²   : {r2:.4f}")


def main():
    dates = get_tgt_dates()
    for date in dates:
        pred_result_path = os.path.join(PRED_RESULT_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        in_situ_path = os.path.join(IN_SITU_TIFF_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        if not os.path.exists(pred_result_path) or not os.path.exists(in_situ_path):
            print(f"No enough data of date: {date}")
            continue
        pre_result_data = read_tiff_data(pred_result_path)
        in_situ_data = read_tiff_data(in_situ_path)
        statistic(pre_result_data, in_situ_data)


if __name__ == "__main__":
    main()
