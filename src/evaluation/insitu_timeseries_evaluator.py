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
from evaluation.heterogeneity_stratified_evaluator import (
    HeterogeneityIndexBuilder,
    GROUP_LABELS,
    build_parent_lookup,
)
from utils.date_util import filter_dates_by_years

PROD_NAMES = [
    DDPM_PIXEL_NAME,
    # ESA_CCI_NAME,
    RF_NAME,
    RESNET_NAME,
    GWR_NAME,
]
RESOLUTION = RESOLUTION_1KM
EVALUATION_DIR_PATH = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_TimeSeries_{RESOLUTION}")
MIN_VALID_DATES = 20
IS_CORRECTION = False
USE_REPRESENTATIVE_SITES = True
REPRESENTATIVE_SITES_PER_GROUP = 2
REPRESENTATIVE_SITE_CSV_NAME = "representative_sites.csv"
EVAL_YEARS = ("2017",)
DISPLAY_NAME_MAP = {
    DDPM_PIXEL_NAME: "SRD-SM",
    RF_NAME: RF_NAME,
    RESNET_NAME: RESNET_NAME,
}


def main():
    dataset = InsituTimeseriesEvalDataset(prod_names=PROD_NAMES, resolution=RESOLUTION)
    dates, data, insitu_indices = dataset.get_all(min_valid_dates=MIN_VALID_DATES)

    os.makedirs(EVALUATION_DIR_PATH, exist_ok=True)

    if USE_REPRESENTATIVE_SITES:
        df_sites = dataset.build_representative_site_table(dates=dates, data=data, insitu_indices=insitu_indices)
        df_sites.to_csv(os.path.join(EVALUATION_DIR_PATH, REPRESENTATIVE_SITE_CSV_NAME), index=False)
        selected_sites = select_representative_sites(
            df_sites=df_sites,
            sites_per_group=REPRESENTATIVE_SITES_PER_GROUP,
        )
    else:
        selected_sites = [{"Row": row, "Col": col, "Group": None} for row, col in insitu_indices]

    for site in selected_sites:
        row, col = int(site["Row"]), int(site["Col"])
        group_label = site.get("Group")
        file_name = f"{row}_{col}.png" if group_label is None else f"{group_label}_{row}_{col}.png"
        out_path = os.path.join(EVALUATION_DIR_PATH, file_name)
        _plot(data, dates, (row, col), out_path, group_label=group_label)

    print(f"Sites: {len(selected_sites)}, dates: {len(dates)}")


class InsituTimeseriesEvalDataset:
    def __init__(self, prod_names, resolution):
        self.prod_names = prod_names
        self.resolution = resolution
        self.insitu_store = InsituStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self._grid_info = self.grid_info_store.get()
        self._coarse_grid_info = GridInfoStore(resolution=RESOLUTION_36KM).get()
        self._parent_ids, _ = build_parent_lookup(
            fine_grid_info=self._grid_info,
            coarse_grid_info=self._coarse_grid_info,
        )
        self._pred_stores = {
            p: data_store_factory.build(p, self.resolution, is_correction=IS_CORRECTION)
            for p in self.prod_names
        }

    @property
    def _dates(self) -> list:
        out = set(self.insitu_store.list_date())
        for store in self._pred_stores.values():
            out &= set(store.list_date())
        return filter_dates_by_years(out, EVAL_YEARS)

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

    def build_representative_site_table(self, dates, data, insitu_indices):
        index_builder = HeterogeneityIndexBuilder(dates=dates)
        group_maps, _ = index_builder.build_group_maps()

        rows = []
        for row, col in insitu_indices:
            ts = data[:, row, col, 0]
            valid_mask = np.isfinite(ts)
            if valid_mask.sum() < MIN_VALID_DATES:
                continue

            group_ids = []
            for i, date in enumerate(dates):
                if not valid_mask[i]:
                    continue
                group_map = group_maps.get(date)
                if group_map is None:
                    continue
                parent_id = int(self._parent_ids[row, col])
                if parent_id < 0:
                    continue
                group_id = int(group_map.reshape(-1)[parent_id])
                if group_id >= 0:
                    group_ids.append(group_id)

            if len(group_ids) < MIN_VALID_DATES:
                continue

            counts = np.bincount(np.asarray(group_ids, dtype=np.int32), minlength=len(GROUP_LABELS))
            dominant_group_id = int(np.argmax(counts))
            rows.append({
                "Row": int(row),
                "Col": int(col),
                "Group": GROUP_LABELS[dominant_group_id],
                "Valid_Dates": int(valid_mask.sum()),
                "Grouped_Dates": int(len(group_ids)),
                "Dominant_Fraction": float(counts[dominant_group_id] / max(len(group_ids), 1)),
                "Low_Count": int(counts[0]),
                "Medium_Count": int(counts[1]),
                "High_Count": int(counts[2]),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "Row", "Col", "Group", "Valid_Dates", "Grouped_Dates",
                "Dominant_Fraction", "Low_Count", "Medium_Count", "High_Count",
            ])

        return pd.DataFrame(rows).sort_values(
            ["Group", "Dominant_Fraction", "Valid_Dates", "Row", "Col"],
            ascending=[True, False, False, True, True],
        ).reset_index(drop=True)


def select_representative_sites(df_sites: pd.DataFrame, sites_per_group: int):
    if df_sites.empty:
        return []

    selected = []
    for group_label in GROUP_LABELS:
        group_df = df_sites[df_sites["Group"] == group_label].copy()
        if group_df.empty:
            continue
        group_df = group_df.sort_values(
            ["Dominant_Fraction", "Valid_Dates", "Grouped_Dates", "Row", "Col"],
            ascending=[False, False, False, True, True],
        )
        selected.extend(group_df.head(sites_per_group).to_dict(orient="records"))
    return selected


def _plot(data, dates, insitu_indices, out_path, group_label=None):
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
        display_name = DISPLAY_NAME_MAP.get(pname, pname)
        r, ubrmse = _calc_metrics(vals, insitu_indices)
        if np.isfinite(r) or np.isfinite(ubrmse):
            r_str = f"{r:.3f}" if np.isfinite(r) else "—"
            u_str = f"{ubrmse:.4f}" if np.isfinite(ubrmse) else "—"
            stats_lines.append(f"{display_name}  R={r_str}  ubRMSE={u_str}")
        valid = np.isfinite(vals)
        if valid.any():
            ax.scatter(
                dates_dt[valid],
                vals[valid],
                label=display_name,
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
    if group_label is None:
        ax.set_title(f"Site (row={rows}, col={cols})")
    else:
        ax.set_title(f"Site (row={rows}, col={cols}, group={group_label})")
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
