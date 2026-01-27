#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Random Forest Trainer
@Author Chris
@Date 2025/12/12
"""
import pickle

import numpy as np
from datasets.rf_dataset import RFTrainDataset
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from constants import *
from evaluation.evaluator import Evaluator

# Random Forest settings
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 42


def main():
    dataset = RFTrainDataset()

    # Collect Data
    X_all, y_all = [], []
    for i in tqdm(range(len(dataset))):
        xs_date = dataset.xs[i]
        ys_date = dataset.ys[i]
        valid_mask = dataset.valid_masks[i]

        # Flatten
        xs_flat = xs_date.reshape(-1, 5)
        ys_flat = ys_date.reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)

        # Filter
        xs_flat = xs_flat[valid_mask_flat]
        ys_flat = ys_flat[valid_mask_flat]

        X_all.append(xs_flat)
        y_all.append(ys_flat)

    # Concatenate
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Train Model
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    rf.fit(X_all, y_all)

    # Evaluate
    pred_all_norm = rf.predict(X_all)
    pred_all = dataset.denorm_y(pred_all_norm)
    y_all_denorm = dataset.denorm_y(y_all)
    metrics = Evaluator.calc_metrics(pred_all, y_all_denorm)
    print(f"  ubRMSE: {metrics['ubRMSE']:.6f}")
    print(f"  R²:     {metrics['R2']:.6f}")
    print(f"  Bias:   {metrics['Bias']:.6f}")
    print(f"  Slope:  {metrics['Slope']:.6f}")

    # Save Model
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf, f)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
