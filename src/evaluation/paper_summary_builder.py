#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Build paper-ready summary tables from evaluation outputs
  @Author Chris
  @Date 2026/3/12
"""
import glob
from typing import List

import pandas as pd
from pandas.errors import EmptyDataError

from constants import *

HETEROGENEITY_RESULT_SUBDIR = f"Evaluation_Heterogeneity_By_Group_{RESOLUTION_1KM}"
SUMMARY_RESULT_SUBDIR = "Evaluation_Paper_Summary"
MAIN_COMPARISON_PRODUCTS = [
    DDPM_PIXEL_NAME,
    RF_NAME,
    RESNET_NAME,
]
ABLATION_PRODUCTS = [DDPM_IMAGE_NAME, DDPM_PIXEL_NAME]
EXISTING_PRODUCT_NAMES = [SM_NAME, ESA_CCI_NAME]
PAPER_PRODUCT_LABELS = {
    DDPM_IMAGE_NAME: "FRD",
    DDPM_PIXEL_NAME: "SRD-SM",
    RF_NAME: RF_NAME,
    RESNET_NAME: RESNET_NAME,
    SM_NAME: "SMAP (36 km)",
    ESA_CCI_NAME: "ESA CCI (36 km)",
}


def main():
    output_dir_path = os.path.join(RESULT_DIR_PATH, SUMMARY_RESULT_SUBDIR)
    os.makedirs(output_dir_path, exist_ok=True)

    df_heterogeneity = build_heterogeneity_overall_summary()
    df_site = build_heterogeneity_site_summary()
    df_stage = build_stage_ablation_summary(df_heterogeneity)
    df_existing = build_existing_product_insitu_summary()
    df_upscale = build_upscale_consistency_summary()

    df_heterogeneity.to_csv(
        os.path.join(output_dir_path, f"Heterogeneity_Overall_Summary_{RESOLUTION_1KM}.csv"),
        index=False,
    )
    df_site.to_csv(
        os.path.join(output_dir_path, f"Heterogeneity_Site_Summary_{RESOLUTION_1KM}.csv"),
        index=False,
    )
    df_stage.to_csv(
        os.path.join(output_dir_path, f"Stage_Ablation_Summary_{RESOLUTION_1KM}.csv"),
        index=False,
    )
    df_existing.to_csv(
        os.path.join(output_dir_path, f"Existing_Product_InSitu_Summary_{RESOLUTION_36KM}.csv"),
        index=False,
    )
    df_upscale.to_csv(
        os.path.join(output_dir_path, "Upscale_Consistency_Summary.csv"),
        index=False,
    )


def build_heterogeneity_overall_summary() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for product_name in MAIN_COMPARISON_PRODUCTS:
        file_path = os.path.join(
            RESULT_DIR_PATH,
            HETEROGENEITY_RESULT_SUBDIR,
            f"{product_name}_overall.csv",
        )
        df = safe_read_csv(file_path)
        if df is None or df.empty:
            continue
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all["Product"] = df_all["Product"].map(PAPER_PRODUCT_LABELS).fillna(df_all["Product"])
    df_all = df_all.sort_values(["Group", "Product"]).reset_index(drop=True)
    return df_all


def build_heterogeneity_site_summary() -> pd.DataFrame:
    rows = []
    for product_name in MAIN_COMPARISON_PRODUCTS:
        file_path = os.path.join(
            RESULT_DIR_PATH,
            HETEROGENEITY_RESULT_SUBDIR,
            f"{product_name}_by_site.csv",
        )
        df = safe_read_csv(file_path)
        if df is None or df.empty:
            continue

        for group_label, group_df in df.groupby("Group"):
            rows.append({
                "Group": group_label,
                "Product": PAPER_PRODUCT_LABELS.get(product_name, product_name),
                "Site_Count": int(len(group_df)),
                "Median_Valid_Dates": float(group_df["Valid_Dates"].median()),
                "Mean_R": float(group_df["Orig_R"].mean()),
                "Median_R": float(group_df["Orig_R"].median()),
                "Mean_ubRMSE": float(group_df["Orig_ubRMSE"].mean()),
                "Median_ubRMSE": float(group_df["Orig_ubRMSE"].median()),
                "Mean_RMSE": float(group_df["Orig_RMSE"].mean()),
                "Median_RMSE": float(group_df["Orig_RMSE"].median()),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Group", "Product"]).reset_index(drop=True)


def build_stage_ablation_summary(df_heterogeneity: pd.DataFrame) -> pd.DataFrame:
    df_stage_rows = []
    for product_name in ABLATION_PRODUCTS:
        file_path = os.path.join(
            RESULT_DIR_PATH,
            HETEROGENEITY_RESULT_SUBDIR,
            f"{product_name}_overall.csv",
        )
        df = safe_read_csv(file_path)
        if df is None or df.empty:
            continue
        df_stage_rows.append(df)

    if not df_stage_rows:
        return pd.DataFrame()

    df_stage_all = pd.concat(df_stage_rows, ignore_index=True)
    df_stage1 = df_stage_all[df_stage_all["Product"] == DDPM_IMAGE_NAME].copy()
    df_final = df_stage_all[df_stage_all["Product"] == DDPM_PIXEL_NAME].copy()
    if df_stage1.empty or df_final.empty:
        return pd.DataFrame()

    merged = df_stage1.merge(
        df_final,
        on="Group",
        suffixes=("_Stage1", "_Final"),
    )
    out = pd.DataFrame({
        "Group": merged["Group"],
        "FRD_R": merged["Orig_R_Stage1"],
        "SRD-SM_R": merged["Orig_R_Final"],
        "Delta_R": merged["Orig_R_Final"] - merged["Orig_R_Stage1"],
        "FRD_ubRMSE": merged["Orig_ubRMSE_Stage1"],
        "SRD-SM_ubRMSE": merged["Orig_ubRMSE_Final"],
        "Delta_ubRMSE": merged["Orig_ubRMSE_Final"] - merged["Orig_ubRMSE_Stage1"],
        "FRD_RMSE": merged["Orig_RMSE_Stage1"],
        "SRD-SM_RMSE": merged["Orig_RMSE_Final"],
        "Delta_RMSE": merged["Orig_RMSE_Final"] - merged["Orig_RMSE_Stage1"],
        "FRD_Valid_Points": merged["Valid_Points_Stage1"],
        "SRD-SM_Valid_Points": merged["Valid_Points_Final"],
    })
    return out.sort_values("Group").reset_index(drop=True)


def build_upscale_consistency_summary() -> pd.DataFrame:
    rows = []
    for by_date_path in iter_upscale_by_date_paths():
        product_name = parse_upscale_product_name(by_date_path)
        if product_name not in MAIN_COMPARISON_PRODUCTS:
            continue

        product_label = PAPER_PRODUCT_LABELS.get(product_name, product_name)
        df_date = safe_read_csv(by_date_path)
        if df_date is None or df_date.empty:
            continue

        row = {
            "Product": product_label,
            "By_Date_Count": int(len(df_date)),
        }
        for col in ["Orig_R", "Orig_ubRMSE", "Orig_Bias"]:
            if col in df_date.columns:
                row[f"By_Date_Mean_{col.replace('Orig_', '')}"] = float(df_date[col].mean())

        by_site_path = build_upscale_by_site_path(by_date_path)
        df_site = safe_read_csv(by_site_path)
        if df_site is not None:
            row["By_Site_Count"] = int(len(df_site))
            for col in ["Orig_R", "Orig_ubRMSE"]:
                if col in df_site.columns:
                    metric_name = col.replace("Orig_", "")
                    row[f"By_Site_Mean_{metric_name}"] = float(df_site[col].mean())
                    row[f"By_Site_Median_{metric_name}"] = float(df_site[col].median())
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Product").reset_index(drop=True)


def build_existing_product_insitu_summary() -> pd.DataFrame:
    rows = []
    by_date_dir_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_By_Date_{RESOLUTION_36KM}")
    by_site_dir_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_Insitu_By_Site_{RESOLUTION_36KM}")

    for product_name in EXISTING_PRODUCT_NAMES:
        df_date = safe_read_csv(os.path.join(by_date_dir_path, f"{product_name}.csv"))
        df_site = safe_read_csv(os.path.join(by_site_dir_path, f"{product_name}.csv"))
        if (df_date is None or df_date.empty) and (df_site is None or df_site.empty):
            continue

        result = {
            "Product": PAPER_PRODUCT_LABELS.get(product_name, product_name),
        }

        if df_date is not None and not df_date.empty:
            result["By_Date_Count"] = int(len(df_date))
            if "Orig_R" in df_date.columns:
                result["By_Date_Mean_R"] = float(df_date["Orig_R"].mean())
            if "Orig_ubRMSE" in df_date.columns:
                result["By_Date_Mean_ubRMSE"] = float(df_date["Orig_ubRMSE"].mean())
            if "Orig_RMSE" in df_date.columns:
                result["By_Date_Mean_RMSE"] = float(df_date["Orig_RMSE"].mean())
            if "Orig_Bias" in df_date.columns:
                result["By_Date_Mean_Bias"] = float(df_date["Orig_Bias"].mean())

        if df_site is not None and not df_site.empty:
            result["By_Site_Count"] = int(len(df_site))
            if "Orig_R" in df_site.columns:
                result["By_Site_Mean_R"] = float(df_site["Orig_R"].mean())
                result["By_Site_Median_R"] = float(df_site["Orig_R"].median())
            if "Orig_ubRMSE" in df_site.columns:
                result["By_Site_Mean_ubRMSE"] = float(df_site["Orig_ubRMSE"].mean())
                result["By_Site_Median_ubRMSE"] = float(df_site["Orig_ubRMSE"].median())
            if "Orig_RMSE" in df_site.columns:
                result["By_Site_Mean_RMSE"] = float(df_site["Orig_RMSE"].mean())
                result["By_Site_Median_RMSE"] = float(df_site["Orig_RMSE"].median())

        rows.append(result)

    return pd.DataFrame(rows).sort_values("Product").reset_index(drop=True)


def iter_upscale_by_date_paths() -> List[str]:
    by_date_dir = os.path.join(RESULT_DIR_PATH, "Evaluation_Upscale_By_Date")
    if os.path.isdir(by_date_dir):
        return sorted(glob.glob(os.path.join(by_date_dir, "*.csv")))
    return sorted(glob.glob(os.path.join(RESULT_DIR_PATH, "Evaluation_Upscale_*_By_Date.csv")))


def build_upscale_by_site_path(by_date_path: str) -> str:
    by_date_dir = os.path.join(RESULT_DIR_PATH, "Evaluation_Upscale_By_Date")
    by_site_dir = os.path.join(RESULT_DIR_PATH, "Evaluation_Upscale_By_Site")
    if by_date_path.startswith(by_date_dir + os.sep):
        return os.path.join(by_site_dir, os.path.basename(by_date_path))
    return by_date_path.replace("_By_Date.csv", "_By_Site.csv")


def parse_upscale_product_name(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    stem = file_name.replace(".csv", "").replace("Evaluation_Upscale_", "").replace("_By_Date", "")
    if stem == "DDPM":
        return "Legacy_DDPM"
    return stem


def safe_read_csv(file_path: str) -> pd.DataFrame | None:
    if not os.path.exists(file_path):
        return None
    try:
        return pd.read_csv(file_path)
    except EmptyDataError:
        return None


if __name__ == "__main__":
    main()
