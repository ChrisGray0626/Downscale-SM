#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Model Predictor
  @Author Chris
  @Date 2025/6/16
"""
import glob

import pandas as pd
from tqdm import tqdm

from Constant import *
from gnnwr import datasets
from trainer import init_model, handle_dataset
from util.tiff_util import interpolate_tiff


def predict_model(pred_data_path):
    pred_data = pd.read_csv(pred_data_path)
    # 加载模型
    train_set, val_set, test_set = handle_dataset(refresh=False)
    gnnwr = init_model(train_set, val_set, test_set)
    gnnwr.load_model(MODEL_FILE_PATH)
    result = []
    # 分批次预测
    total = len(pred_data)
    batch_size = 20000
    for start in tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch_pred_data = pred_data[start:end].copy()
        batch_pred_dataset = datasets.init_predict_dataset(data=batch_pred_data, train_dataset=train_set,
                                                           x_column=X_COLUMN,
                                                           spatial_column=SPATIAL_COLUMN,
                                                           )
        batch_result = gnnwr.predict(batch_pred_dataset)
        result.append(batch_result)
    result_df = pd.concat(result, ignore_index=True)

    return result_df


def handle_pred_result(pred_result):
    date = pred_result[DATE_NAME][0]
    lon = pred_result[LONGITUDE_NAME]
    lat = pred_result[LATITUDE_NAME]
    data = pred_result['pred_result']
    dst_path = os.path.join(PRED_RESULT_DIR_PATH, f"{date}{TIFF_SUFFIX}")
    os.makedirs(PRED_RESULT_DIR_PATH, exist_ok=True)
    interpolate_tiff(data, lon, lat, REF_GRID_1KM_PATH, dst_path)
    csv_path = os.path.join(PRED_RESULT_DIR_PATH, f"{date}{CSV_SUFFIX}")
    pred_result.to_csv(csv_path, index=False)


def main():
    file_paths = glob.glob(os.path.join(PRED_DATA_DIR_PATH, f"*{CSV_SUFFIX}"))
    for file_path in tqdm(file_paths):
        pred_result = predict_model(file_path)
        handle_pred_result(pred_result)


if __name__ == "__main__":
    main()
