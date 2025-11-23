#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle Albedo Data
  @Author Chris
  @Date 2025/5/19
"""

from constant import *
from data.NDVIHandler import BatchConvert2TiffJob, BatchMergeTiffJob, BatchMultiResampleTiffJob, ResolutionConfig
from util.workflow.Base import *

# Gap Valuer
GAP_VALUE = 32766
SCALE_FACTOR = 0.001
# TODO
DIR_NAME = ALBEDO_NAME
SRC_DIR_PATH = os.path.join(RAW_DIR_PATH, DIR_NAME)
TIFF_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, CONVERTED_DIR_NAME)
MERGED_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, MERGED_DIR_NAME)
DST_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME)


def main():
    job = Job()
    context = Context()

    job.add([
        BatchConvert2TiffJob(),
        BatchMergeTiffJob(),
        BatchMultiResampleTiffJob(),
    ])

    # Convert to TIFF Config
    # Read Config
    context.set_global(RAW_DIR_PATH_KEY, SRC_DIR_PATH)
    # MODIS Data Process Config
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    context.set_global(SCALE_FACTOR_KEY, SCALE_FACTOR)
    # Write Config
    context.set_global(CONVERTED_DIR_PATH_KEY, TIFF_DIR_PATH)

    # Merge Tiff Config
    context.set_global(MERGED_DIR_PATH_KEY, MERGED_DIR_PATH)

    # Multi-Resolution Resample Config
    # Multi Resample Config
    context.set_global(RESOLUTION_CONFIGS_KEY, [
        ResolutionConfig(
            resolution_km=1,
            ref_grid_path=STANDARD_GRID_1KM_PATH,
        ),
        ResolutionConfig(
            resolution_km=36,
            ref_grid_path=STANDARD_GRID_36KM_PATH,
        ),
    ])
    # Resample Config
    context.set_global(RESAMPLED_DIR_PATH_KEY, DST_DIR_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
