#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Model Trainer
  @Author Chris
  @Date 2025/6/13
"""
import glob

import pandas as pd
from torch import nn

from Constant import *
from gnnwr import datasets, models


def handle_input_data():
    input_data_files = glob.glob(os.path.join(LABELED_DATA_DIR_PATH, f'*{CSV_SUFFIX}'))
    input_data = pd.concat([pd.read_csv(f) for f in input_data_files], ignore_index=True)

    return input_data


def handle_dataset(refresh=False):
    if refresh is False and os.path.exists(DATASET_DIR_PATH):
        train_set = datasets.load_dataset(TRAIN_DATASET_DIR_PATH)
        val_set = datasets.load_dataset(VAL_DATASET_DIR_PATH)
        test_set = datasets.load_dataset(TEST_DATASET_DIR_PATH)
        return train_set, val_set, test_set

    input_data = handle_input_data()
    # 混洗数据
    input_data = input_data.sample(frac=1).reset_index(drop=True)
    # 分割数据
    train_data = input_data[:int(0.8 * len(input_data))]
    val_data = input_data[int(0.8 * len(input_data)):int(0.9 * len(input_data))]
    test_data = input_data[int(0.9 * len(input_data)):]
    train_set, val_set, test_set = datasets.init_dataset_usedata(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        x_column=X_COLUMN,
        y_column=Y_COLUMN,
        spatial_column=SPATIAL_COLUMN,
        use_model="gnnwr")
    # 存储数据集
    train_set.save(TRAIN_DATASET_DIR_PATH, exist_ok=True)
    val_set.save(VAL_DATASET_DIR_PATH, exist_ok=True)
    test_set.save(TEST_DATASET_DIR_PATH, exist_ok=True)

    return train_set, val_set, test_set


def init_model(train_set, val_set, test_set):
    optimizer_params = {
        "scheduler": "MultiStepLR",
        "scheduler_milestones": [500, 1000, 1500, 2000],
        "scheduler_gamma": 0.75,
    }
    gnnwr = models.GNNWR(train_dataset=train_set,
                         valid_dataset=val_set,
                         test_dataset=test_set,
                         dense_layers=[1024, 512, 256],
                         activate_func=nn.PReLU(init=0.4),
                         start_lr=0.02,
                         optimizer="Adadelta",
                         model_name=MODEL_NAME,
                         model_save_path=MODEL_DIR_PATH,
                         log_path=os.path.join(GNNWR_DIR, "Log"),
                         write_path=os.path.join(GNNWR_DIR, "Run"),
                         use_gpu=False,
                         optimizer_params=optimizer_params
                         )

    return gnnwr


def main():
    train_set, val_set, test_set = handle_dataset(refresh=True)
    gnnwr = init_model(train_set, val_set, test_set)
    gnnwr.run(max_epoch=100, early_stop=50)


if __name__ == "__main__":
    main()
