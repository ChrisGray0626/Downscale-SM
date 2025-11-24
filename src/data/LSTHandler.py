#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle LST Data
  @Author Chris
  @Date 2025/5/5
"""


from Constant import *
from util.workflow.Base import Job, Context
from util.workflow.Job import BatchMODISData2TiffJob, BatchMergeTiffJob, BatchMultiResampleTiffJob, ResolutionConfig
from util.workflow.WorkflowConstant import *
# Gap Value
GAP_VALUE = 0
SCALE_FACTOR = 0.02

DATA_NAME = LST_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, CONVERTED_DIR_NAME)
MERGED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, MERGED_DIR_NAME)
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME)


def main():
    job = Job()
    context = Context()

    job.add([
        BatchMODISData2TiffJob(
            src_dir_path_key=RAW_DIR_PATH_KEY,
            dst_dir_path_key=CONVERTED_DIR_PATH_KEY,
            data_name=DATA_NAME,
            gap_value_key=GAP_VALUE_KEY,
            scale_factor_key=SCALE_FACTOR_KEY
        ),
        BatchMergeTiffJob(
            src_dir_path_key=CONVERTED_DIR_PATH_KEY,
            dst_dir_path_key=MERGED_DIR_PATH_KEY
        ),
        BatchMultiResampleTiffJob(
            resolution_configs_key=RESOLUTION_CONFIGS_KEY,
            src_dir_path_key=MERGED_DIR_PATH_KEY,
            ref_grid_path_key=REF_GRID_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY,
        ),
    ])

    # Convert to TIFF Config
    # Read Config
    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    # MODIS Data Process Config
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    context.set_global(SCALE_FACTOR_KEY, SCALE_FACTOR)
    # Write Config
    context.set_global(CONVERTED_DIR_PATH_KEY, CONVERTED_DIR_PATH)

    # Merge Tiff Config
    context.set_global(MERGED_DIR_PATH_KEY, MERGED_DIR_PATH)

    # Multi-Resolution Resample Config
    # Multi Resample Config
    context.set_global(RESOLUTION_CONFIGS_KEY, [
        ResolutionConfig(
            resolution_km=1,
            ref_grid_path=REF_GRID_1KM_PATH,
        ),
        ResolutionConfig(
            resolution_km=36,
            ref_grid_path=REF_GRID_36KM_PATH,
        ),
    ])
    # Resample Config
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
