#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle DEM Data
  @Author Chris
  @Date 2025/5/6
"""

from Constant import *
from util.workflow.common.Decompress import BatchDecompressJob
from util.workflow.common.Merge import TiffMerger
from util.workflow.common.Resample import MultiResampleTiffJob, ResolutionConfig
from util.workflow.core.Base import Job, Context

from util.workflow.core.ContextKey import *

DATA_NAME = DEM_NAME
RAW_DIR_PATH = os.path.join(RAW_DIR_PATH, DATA_NAME)
DECOMPRESSED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, DECOMPRESSED_DIR_NAME)
MERGED_FILE_PATH = os.path.join(PROCESSED_DIR_PATH, DATA_NAME, MERGED_DIR_NAME, f"{DATA_NAME}{TIFF_SUFFIX}")
RESAMPLED_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, DEM_NAME)


def main():
    job = Job()
    context = Context()

    job.add([
        BatchDecompressJob(
            src_dir_path_key=RAW_DIR_PATH_KEY,
            dst_dir_path_key=DECOMPRESSED_DIR_PATH_KEY
        ),
        TiffMerger(
            src_dir_path_key=DECOMPRESSED_DIR_PATH_KEY,
            dst_file_path_key=MERGED_DIR_PATH_KEY
        ),
        MultiResampleTiffJob(
            resolution_configs_key=RESOLUTION_CONFIGS_KEY,
            src_file_path_key=MERGED_DIR_PATH_KEY,
            ref_grid_path_key=REF_GRID_PATH_KEY,
            dst_dir_path_key=RESAMPLED_DIR_PATH_KEY
        ),
    ])

    # Decompress Config
    context.set_global(RAW_DIR_PATH_KEY, RAW_DIR_PATH)
    context.set_global(DECOMPRESSED_DIR_PATH_KEY, DECOMPRESSED_DIR_PATH)

    # Merge Config
    context.set_global(MERGED_DIR_PATH_KEY, MERGED_FILE_PATH)

    # Multi-Resolution Resample Config
    context.set_global(MERGED_DIR_PATH_KEY, MERGED_FILE_PATH)
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
    context.set_global(RESAMPLED_DIR_PATH_KEY, RESAMPLED_DIR_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
