#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of In-situ Time Series
  @Author Chris
  @Date 2026/1/27
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *
from datasets import data_store_factory
from datasets.common_data_store import GridInfoStore, InsituStore
from evaluation.evaluator import Evaluator

PROD_NAMES = [
    DDPM_PIXEL_NAME,
    DDPM_IMAGE_NAME,
    # ESA_CCI_NAME,
    RF_NAME,
    RESNET_NAME,
]
RESOLUTION = RESOLUTION_1KM
EVALUATION_DIR_PATH = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_TimeSeries_{RESOLUTION}")
MIN_VALID_DATES = 20


def main():
    dataset = InsituTimeseriesEvalDataset(prod_names=PROD_NAMES, resolution=RESOLUTION)
    dates, data, insitu_indices = dataset.get_all(min_valid_dates=MIN_VALID_DATES)

    os.makedirs(EVALUATION_DIR_PATH, exist_ok=True)
    for row, col in insitu_indices:
        out_path = os.path.join(EVALUATION_DIR_PATH, f"{row}_{col}.png")
        _plot(data, dates, (row, col), out_path)

    print(f"Sites: {len(insitu_indices)}, dates: {len(dates)}")


class InsituTimeseriesEvalDataset:
    def __init__(self, prod_names, resolution):
        self.prod_names = prod_names
        self.resolution = resolution
        self.insitu_store = InsituStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._grid_info = self.grid_info_store.get()
        self._pred_stores = {
            p: data_store_factory.build(p, self.resolution, is_correction=True)
            for p in self.prod_names
        }

    @property
    def _dates(self) -> list:
        out = set(self.insitu_store.list_date())
        for store in self._pred_stores.values():
            out &= set(store.list_date())
        return sorted(out)

    def get_all(self, min_valid_dates=MIN_VALID_DATES):
        grid_info = self._grid_info
        H, W = grid_info["H"], grid_info["W"]

        dates = self._dates
        data = np.full((len(dates), H, W, 1 + len(self.prod_names)), np.nan, dtype=np.float32)  # [D, H, W, P]
        for i, date in enumerate(dates):
            insitu = self.insitu_store.get(date).astype(np.float32)
            data[i, :, :, 0] = np.where(np.isfinite(insitu), insitu, np.nan)
            for j, prod_name in enumerate(self.prod_names):
                arr = self._pred_stores[prod_name].get(date).astype(np.float32)
                data[i, :, :, 1 + j] = np.where(np.isfinite(arr), arr, np.nan)

        insitu_valid_masks = np.isfinite(data[:, :, :, 0]).sum(axis=0)
        insitu_indices = [(int(r), int(c)) for r in range(H) for c in range(W) if
                          insitu_valid_masks[r, c] >= min_valid_dates]

        return dates, data, insitu_indices


def _plot(data, dates, insitu_indices, out_path):
    D, H, W, P = data.shape
    rows, cols = insitu_indices
    if rows < 0 or rows >= H or cols < 0 or cols >= W:
        return

    ts = data[:, rows, cols, :]
    insitu_indices = ts[:, 0]
    dates_dt = pd.to_datetime(dates, format="%Y%m%d")

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(top=0.82)
    ax.plot(dates_dt, insitu_indices, color="black", label="In-situ", linestyle="-", linewidth=1.2, zorder=3)

    markers = ["o", "s", "^"]
    colors = ["C0", "C1", "C2"]
    stats_lines = []
    for idx, pname in enumerate(PROD_NAMES):
        vals = ts[:, 1 + idx]
        r, ubrmse = _calc_metrics(vals, insitu_indices)
        if np.isfinite(r) or np.isfinite(ubrmse):
            r_str = f"{r:.3f}" if np.isfinite(r) else "—"
            u_str = f"{ubrmse:.4f}" if np.isfinite(ubrmse) else "—"
            stats_lines.append(f"{pname}  R={r_str}  ubRMSE={u_str}")
        valid = np.isfinite(vals)
        if valid.any():
            ax.scatter(
                dates_dt[valid],
                vals[valid],
                label=pname,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                alpha=0.8,
                s=18,
                zorder=2,
            )

    if stats_lines:
        fig.text(0.02, 0.96, "\n".join(stats_lines), ha="left", va="top",
                 fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

    ax.set_xlabel("Date")
    ax.set_ylabel("Soil moisture")
    ax.set_title(f"Site (row={rows}, col={cols})")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _calc_metrics(pred, true):
    valid = (np.isfinite(pred) & np.isfinite(true)).astype(np.float32)
    return Evaluator.r(pred, true, valid), Evaluator.ubrmse(pred, true, valid)


if __name__ == "__main__":
    main()
