#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Random Forest Trainer
@Author Chris
@Date 2025/12/12
"""
import pickle

from sklearn.ensemble import RandomForestRegressor

from constants import *
from evaluation.evaluator import Evaluator
from rf_dataset import RFTrainDataset

# Random Forest settings
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 42


def main():
    dataset = RFTrainDataset()
    xs, ys = dataset.get_all()

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

    rf.fit(xs, ys)

    # Evaluate
    pred_all_norm = rf.predict(xs)
    pred_all = dataset.denorm_y(pred_all_norm)
    y_all_denorm = dataset.denorm_y(ys)
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
