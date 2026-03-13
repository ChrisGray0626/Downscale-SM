#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Evaluation of 1km products under coarse-cell heterogeneity stratification
  @Author Chris
  @Date 2026/3/10
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rasterio.warp import reproject, Resampling

from constants import *
from datasets import data_store_factory
from datasets.common_data_store import GridInfoStore, InsituStore, ModelDataStore
from evaluation.evaluator import Evaluator
from utils.date_util import filter_dates_by_years, get_valid_dates

PROD_NAMES = [
    DDPM_IMAGE_NAME,
    DDPM_PIXEL_NAME,
    RF_NAME,
    RESNET_NAME,
]

FINE_RESOLUTION = RESOLUTION_1KM
COARSE_RESOLUTION = RESOLUTION_36KM
HETEROGENEITY_FEAT_NAMES = [NDVI_NAME, LST_NAME, DEM_NAME]
MIN_SUBCELL_VALID_RATIO = 0.5
MIN_INDEX_COMPONENT_NUM = 2
GROUP_LABELS = ("Low", "Medium", "High")
RESULT_SUBDIR = f"Evaluation_Heterogeneity_By_Group_{FINE_RESOLUTION}"
MIN_GROUP_DATE_POINTS = 5
MIN_SITE_GROUP_DATES = 5
IS_CORRECTION = False
EVAL_YEARS = ("2017",)


def main():
    evaluator = Evaluator(min_site_num=2, min_date_num=2)
    output_dir_path = os.path.join(RESULT_DIR_PATH, RESULT_SUBDIR)
    os.makedirs(output_dir_path, exist_ok=True)

    insitu_store = InsituStore(resolution=FINE_RESOLUTION)
    common_dates = filter_dates_by_years(set(get_valid_dates()) & set(insitu_store.list_date()), EVAL_YEARS)

    index_builder = HeterogeneityIndexBuilder(dates=common_dates)
    group_maps, df_group_info = index_builder.build_group_maps()
    df_group_info.to_csv(
        os.path.join(output_dir_path, "heterogeneity_group_by_date.csv"),
        index=False,
    )

    for product_name in PROD_NAMES:
        dataset = HeterogeneityStratifiedEvalDataset(
            product_name=product_name,
            dates=common_dates,
            group_maps=group_maps,
        )
        df_overall, df_by_date, df_by_site = dataset.evaluate(evaluator=evaluator)
        df_overall.to_csv(
            os.path.join(output_dir_path, f"{product_name}_overall.csv"),
            index=False,
        )
        df_by_date.to_csv(
            os.path.join(output_dir_path, f"{product_name}_by_date.csv"),
            index=False,
        )
        df_by_site.to_csv(
            os.path.join(output_dir_path, f"{product_name}_by_site.csv"),
            index=False,
        )


class HeterogeneityIndexBuilder:
    def __init__(self, dates: List[str]):
        self.dates = dates
        self.data_store = ModelDataStore(resolution=FINE_RESOLUTION)
        self.fine_grid_info = GridInfoStore(resolution=FINE_RESOLUTION).get()
        self.coarse_grid_info = GridInfoStore(resolution=COARSE_RESOLUTION).get()
        self.parent_ids, self.expected_counts = build_parent_lookup(
            fine_grid_info=self.fine_grid_info,
            coarse_grid_info=self.coarse_grid_info,
        )
        self.coarse_shape = (self.coarse_grid_info["H"], self.coarse_grid_info["W"])
        self.dem_norm = self._build_dem_heterogeneity()

    def build_group_maps(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        group_maps: Dict[str, np.ndarray] = {}
        rows = []
        for date in self.dates:
            ndvi_std = calc_parent_std_map(
                fine_data=self.data_store.get(NDVI_NAME, date),
                parent_ids=self.parent_ids,
                expected_counts=self.expected_counts,
                coarse_shape=self.coarse_shape,
            )
            lst_std = calc_parent_std_map(
                fine_data=self.data_store.get(LST_NAME, date),
                parent_ids=self.parent_ids,
                expected_counts=self.expected_counts,
                coarse_shape=self.coarse_shape,
            )
            ndvi_norm = minmax_norm(ndvi_std)
            lst_norm = minmax_norm(lst_std)
            index_map, component_count = build_composite_index(
                [ndvi_norm, lst_norm, self.dem_norm],
                min_component_num=MIN_INDEX_COMPONENT_NUM,
            )
            group_map, thresholds = stratify_index_map(index_map)
            group_maps[date] = group_map
            valid_group = group_map >= 0
            rows.append({
                DATE_NAME: date,
                "Valid_Cells": int(np.isfinite(index_map).sum()),
                "Valid_Grouped_Cells": int(valid_group.sum()),
                "Low_Cells": int((group_map == 0).sum()),
                "Medium_Cells": int((group_map == 1).sum()),
                "High_Cells": int((group_map == 2).sum()),
                "Q33": thresholds[0],
                "Q67": thresholds[1],
                "Index_Mean": float(np.nanmean(index_map)),
                "Index_Std": float(np.nanstd(index_map)),
                "Mean_Component_Num": float(np.nanmean(component_count)),
            })
        return group_maps, pd.DataFrame(rows)

    def _build_dem_heterogeneity(self) -> np.ndarray:
        dem = self.data_store.get(DEM_NAME)
        dem_std = calc_parent_std_map(
            fine_data=dem,
            parent_ids=self.parent_ids,
            expected_counts=self.expected_counts,
            coarse_shape=self.coarse_shape,
        )
        return minmax_norm(dem_std)


class HeterogeneityStratifiedEvalDataset:
    def __init__(self, product_name: str, dates: List[str], group_maps: Dict[str, np.ndarray]):
        self.product_name = product_name
        self.pred_store = data_store_factory.build(product_name, FINE_RESOLUTION, is_correction=IS_CORRECTION)
        self.insitu_store = InsituStore(resolution=FINE_RESOLUTION)
        self.group_maps = group_maps
        self.dates = sorted(
            set(dates)
            & set(self.pred_store.list_date())
            & set(self.insitu_store.list_date())
            & set(self.group_maps.keys())
        )
        self.fine_grid_info = GridInfoStore(resolution=FINE_RESOLUTION).get()
        self.coarse_grid_info = GridInfoStore(resolution=COARSE_RESOLUTION).get()
        self.parent_ids, _ = build_parent_lookup(
            fine_grid_info=self.fine_grid_info,
            coarse_grid_info=self.coarse_grid_info,
        )
        self.row_grid = self.fine_grid_info["rows"].astype(np.int32)
        self.col_grid = self.fine_grid_info["cols"].astype(np.int32)

    def evaluate(self, evaluator: Evaluator) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        overall_rows = []
        by_date_rows = []
        records = {
            label: {"pred": [], "true": [], "date": [], "row": [], "col": []}
            for label in GROUP_LABELS
        }

        for date in self.dates:
            pred_map = self.pred_store.get(date).astype(np.float32)
            true_map = self.insitu_store.get(date).astype(np.float32)
            group_map = self.group_maps[date].reshape(-1)[self.parent_ids.reshape(-1)].reshape(self.parent_ids.shape)
            valid = np.isfinite(pred_map) & np.isfinite(true_map) & (group_map >= 0)
            if valid.sum() == 0:
                continue

            pred_valid = pred_map[valid]
            true_valid = true_map[valid]
            group_valid = group_map[valid]
            row_valid = self.row_grid[valid]
            col_valid = self.col_grid[valid]

            for group_id, label in enumerate(GROUP_LABELS):
                group_mask = group_valid == group_id
                group_pred = pred_valid[group_mask]
                group_true = true_valid[group_mask]
                if group_pred.size == 0:
                    continue

                records[label]["pred"].append(group_pred)
                records[label]["true"].append(group_true)
                records[label]["date"].append(np.full(group_pred.shape[0], date, dtype=object))
                records[label]["row"].append(row_valid[group_mask])
                records[label]["col"].append(col_valid[group_mask])

                if group_pred.size >= MIN_GROUP_DATE_POINTS:
                    metrics = build_metric_rows(evaluator, group_pred, group_true)
                    by_date_rows.append({
                        DATE_NAME: date,
                        "Group": label,
                        "Product": self.product_name,
                        "Valid_Points": int(group_pred.size),
                        **metrics,
                    })

        for label in GROUP_LABELS:
            if not records[label]["pred"]:
                continue
            pred = np.concatenate(records[label]["pred"]).astype(np.float32)
            true = np.concatenate(records[label]["true"]).astype(np.float32)
            dates = np.concatenate(records[label]["date"])
            metrics = build_metric_rows(evaluator, pred, true)
            overall_rows.append({
                "Group": label,
                "Product": self.product_name,
                "Valid_Points": int(pred.size),
                "Valid_Dates": int(len(np.unique(dates))),
                **metrics,
            })

        df_overall = pd.DataFrame(overall_rows)
        df_by_date = pd.DataFrame(by_date_rows)
        df_by_site = self._build_by_site_df(records=records, evaluator=evaluator)
        if not df_overall.empty:
            df_overall = df_overall.sort_values(["Group"])
        if not df_by_date.empty:
            df_by_date = df_by_date.sort_values([DATE_NAME, "Group"])
        if not df_by_site.empty:
            df_by_site = df_by_site.sort_values(["Group", "Corr_R"], ascending=[True, False], na_position="last")
        return df_overall, df_by_date, df_by_site

    def _build_by_site_df(self, records: Dict[str, dict], evaluator: Evaluator) -> pd.DataFrame:
        rows = []
        for label in GROUP_LABELS:
            if not records[label]["pred"]:
                continue

            df = pd.DataFrame({
                "Row": np.concatenate(records[label]["row"]).astype(np.int32),
                "Col": np.concatenate(records[label]["col"]).astype(np.int32),
                DATE_NAME: np.concatenate(records[label]["date"]),
                "Pred": np.concatenate(records[label]["pred"]).astype(np.float32),
                "True": np.concatenate(records[label]["true"]).astype(np.float32),
            })
            if df.empty:
                continue

            for (row, col), group in df.groupby(["Row", "Col"]):
                pred = group["Pred"].values.astype(np.float32)
                true = group["True"].values.astype(np.float32)
                valid_dates = int(group[DATE_NAME].nunique())
                if valid_dates < MIN_SITE_GROUP_DATES:
                    continue
                metrics = build_metric_rows(evaluator, pred, true)
                rows.append({
                    "Row": int(row),
                    "Col": int(col),
                    "Group": label,
                    "Product": self.product_name,
                    "Valid_Dates": valid_dates,
                    **metrics,
                })

        return pd.DataFrame(rows)


def build_parent_lookup(fine_grid_info: dict, coarse_grid_info: dict) -> Tuple[np.ndarray, np.ndarray]:
    coarse_h = coarse_grid_info["H"]
    coarse_w = coarse_grid_info["W"]
    coarse_ids = np.arange(coarse_h * coarse_w, dtype=np.float32).reshape(coarse_h, coarse_w)
    fine_parent_ids = np.full((fine_grid_info["H"], fine_grid_info["W"]), -1, dtype=np.float32)
    reproject(
        source=coarse_ids,
        destination=fine_parent_ids,
        src_transform=coarse_grid_info["transform"],
        src_crs=coarse_grid_info["crs"],
        dst_transform=fine_grid_info["transform"],
        dst_crs=fine_grid_info["crs"],
        resampling=Resampling.nearest,
    )
    fine_parent_ids = fine_parent_ids.astype(np.int32)
    expected_counts = np.bincount(
        fine_parent_ids.reshape(-1),
        minlength=coarse_h * coarse_w,
    ).astype(np.float32)
    return fine_parent_ids, expected_counts


def calc_parent_std_map(
        fine_data: np.ndarray,
        parent_ids: np.ndarray,
        expected_counts: np.ndarray,
        coarse_shape: Tuple[int, int],
) -> np.ndarray:
    flat_data = fine_data.reshape(-1).astype(np.float64)
    flat_ids = parent_ids.reshape(-1)
    valid = np.isfinite(flat_data) & (flat_ids >= 0)

    coarse_size = coarse_shape[0] * coarse_shape[1]
    valid_counts = np.bincount(flat_ids[valid], minlength=coarse_size).astype(np.float64)
    value_sum = np.bincount(flat_ids[valid], weights=flat_data[valid], minlength=coarse_size)
    value_sq_sum = np.bincount(flat_ids[valid], weights=flat_data[valid] ** 2, minlength=coarse_size)

    mean = np.divide(value_sum, valid_counts, out=np.full(coarse_size, np.nan), where=valid_counts > 0)
    variance = np.divide(value_sq_sum, valid_counts, out=np.full(coarse_size, np.nan),
                         where=valid_counts > 0) - mean ** 2
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)

    coverage = np.divide(
        valid_counts,
        expected_counts,
        out=np.zeros(coarse_size, dtype=np.float64),
        where=expected_counts > 0,
    )
    std[(coverage < MIN_SUBCELL_VALID_RATIO) | (valid_counts < 2)] = np.nan
    return std.reshape(coarse_shape).astype(np.float32)


def minmax_norm(data: np.ndarray) -> np.ndarray:
    out = np.full_like(data, np.nan, dtype=np.float32)
    valid = np.isfinite(data)
    if not np.any(valid):
        return out
    valid_data = data[valid].astype(np.float64)
    data_min = valid_data.min()
    data_max = valid_data.max()
    if np.isclose(data_max, data_min):
        out[valid] = 0.0
        return out
    out[valid] = ((valid_data - data_min) / (data_max - data_min)).astype(np.float32)
    return out


def build_composite_index(component_maps: List[np.ndarray], min_component_num: int) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(component_maps, axis=0).astype(np.float32)
    valid = np.isfinite(stacked)
    component_count = valid.sum(axis=0).astype(np.int32)
    index_map = np.full(stacked.shape[1:], np.nan, dtype=np.float32)
    enough = component_count >= min_component_num
    if np.any(enough):
        index_map[enough] = np.nanmean(stacked[:, enough], axis=0).astype(np.float32)
    return index_map, component_count


def stratify_index_map(index_map: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    group_map = np.full(index_map.shape, -1, dtype=np.int8)
    valid = np.isfinite(index_map)
    if valid.sum() < len(GROUP_LABELS):
        return group_map, (np.nan, np.nan)
    values = index_map[valid].astype(np.float64)
    q33, q67 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    group_map[valid & (index_map <= q33)] = 0
    group_map[valid & (index_map > q33) & (index_map <= q67)] = 1
    group_map[valid & (index_map > q67)] = 2
    return group_map, (float(q33), float(q67))


def build_metric_rows(evaluator: Evaluator, pred: np.ndarray, true: np.ndarray) -> dict:
    if pred.size < 2 or true.size < 2:
        return {
            "Orig_ubRMSE": np.nan,
            "Orig_Bias": np.nan,
            "Orig_R": np.nan,
            "Orig_Slope": np.nan,
            "Orig_RMSE": np.nan,
            "Corr_ubRMSE": np.nan,
            "Corr_Bias": np.nan,
            "Corr_R": np.nan,
            "Corr_Slope": np.nan,
            "Corr_RMSE": np.nan,
            "Systematic_Bias": np.nan,
        }

    orig_metrics = evaluator.calc_metrics(pred, true)
    orig_rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    systematic_bias = float(np.mean(pred - true))
    pred_corrected = pred - systematic_bias
    corr_metrics = evaluator.calc_metrics(pred_corrected, true)
    corr_rmse = float(np.sqrt(np.mean((pred_corrected - true) ** 2)))
    return {
        "Orig_ubRMSE": orig_metrics["ubRMSE"],
        "Orig_Bias": orig_metrics["Bias"],
        "Orig_R": orig_metrics["R"],
        "Orig_Slope": orig_metrics["Slope"],
        "Orig_RMSE": orig_rmse,
        "Corr_ubRMSE": corr_metrics["ubRMSE"],
        "Corr_Bias": corr_metrics["Bias"],
        "Corr_R": corr_metrics["R"],
        "Corr_Slope": corr_metrics["Slope"],
        "Corr_RMSE": corr_rmse,
        "Systematic_Bias": systematic_bias,
    }


if __name__ == "__main__":
    main()
