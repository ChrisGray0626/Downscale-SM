#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Geographically Weighted Regression Trainer
@Author Chris
@Date 2026/1/27
"""

import pickle

import joblib
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from constants import *
from evaluation.evaluator import Evaluator
from gwr.gwr_dataset import GWRTrainDataset
from utils.util import suppress_linalg

MAX_SAMPLE_BW = 30000


def main():
    dataset = GWRTrainDataset()
    lons, lats, xs, ys = dataset.get_all()

    # bw = search_bw(lons, lats, ys, xs)
    bw = 238.0

    pos_full_list = list(zip(lons, lats))
    model = GWR(
        pos_full_list,
        ys,
        xs,
        bw=bw,
        constant=True,
        spherical=True,
        n_jobs=-1,
    )
    with joblib.parallel_backend("loky", initializer=suppress_linalg, initargs=()):  # type: ignore[call-arg]
        results = model.fit()
    pred_norm = results.predy
    pred = dataset.denorm_y(pred_norm.flatten())
    y_denorm = dataset.denorm_y(ys.flatten())
    metrics = Evaluator.calc_metrics(pred, y_denorm)
    print(f"  ubRMSE: {metrics['ubRMSE']:.6f}")
    print(f"  R²:     {metrics['R2']:.6f}")
    print(f"  Bias:   {metrics['Bias']:.6f}")
    print(f"  Slope:  {metrics['Slope']:.6f}")

    os.makedirs(os.path.dirname(GWR_MODEL_PATH), exist_ok=True)
    with open(GWR_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)  # type: ignore[arg-type]


def search_bw(lons, lats, ys, xs):
    if ys.ndim == 1:
        ys = ys.reshape(-1, 1)

    n = len(ys)
    if n > MAX_SAMPLE_BW:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=MAX_SAMPLE_BW, replace=False)
        lons_bw = lons[idx]
        lats_bw = lats[idx]
        y_bw = ys[idx]
        X_bw = xs[idx]
    else:
        lons_bw, lats_bw, y_bw, X_bw = lons, lats, ys, xs

    pos_list = list(zip(lons_bw, lats_bw))

    sel = Sel_BW(
        pos_list,
        y_bw,
        X_bw,
        constant=True,
        spherical=True,
    )
    bw = sel.search(criterion="AICc")
    print(f"Selected bandwidth: {bw}")


if __name__ == "__main__":
    main()
