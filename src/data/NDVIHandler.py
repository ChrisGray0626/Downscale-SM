#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle NDVI Data
  @Author Chris
  @Date 2025/5/5
"""

from util.workflow.Job import *

# Gap Value
GAP_VALUE = -3000
SCALE_FACTOR = 0.0001

DIR_NAME = NDVI_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DIR_NAME)
CONVERTED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DIR_NAME, CONVERTED_DIR_NAME)
MERGED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DIR_NAME, MERGED_DIR_NAME)
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DIR_NAME)


def main():
    job = Job()
    context = Context()

    job.add([
        BatchMODISData2TiffJob(),
        BatchMergeTiffJob(),
        BatchMultiResampleTiffJob(),
        ValidDateHandler(),
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
