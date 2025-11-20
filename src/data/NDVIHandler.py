#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle NDVI Data
  @Author Chris
  @Date 2025/5/5
"""
import glob
from collections import defaultdict

from constant import *
from util.util import extract_date_from_modis_filename
from util.workflow.Base import *
from util.workflow.Task import HDF4Reader, TiffWriter, MODISDataProcessor, TiffMerger, TiffResampler
from dataclasses import dataclass

# Gap Value
GAP_VALUE = -3000
SCALE_FACTOR = 0.0001

DIR_NAME = NDVI_NAME
INPUT_DIR_PATH = os.path.join(DATA_PATH, DIR_NAME)
TIFF_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, TIFF_DIR_NAME)
MERGED_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME, MERGED_DIR_NAME)
DST_DIR_PATH = os.path.join(RESULT_PATH, DIR_NAME)


@dataclass
class ResolutionConfig:
    resolution_km: int
    ref_grid_path: str


class BatchConvert2TiffJob(BatchJob):

    def __init__(self):
        super().__init__()
        self.add(
            HDF4Reader(),
            MODISDataProcessor(),
            TiffWriter())

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(SRC_DIR_PATH_KEY)
        tiff_dir_path = context.get(TIFF_DIR_PATH_KEY)

        os.makedirs(tiff_dir_path, exist_ok=True)

        batch_contexts = []
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{HDF4_SUFFIX}"))
        for src_file_path in src_file_paths:
            filename = os.path.basename(src_file_path)
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)

            dst_path = os.path.join(tiff_dir_path, filename.replace(HDF4_SUFFIX, TIFF_SUFFIX))
            batch_context.set(DST_FILE_PATH_KEY, dst_path)

            batch_contexts.append(batch_context)

        return batch_contexts


class BatchMergeTiffJob(BatchJob):

    def __init__(self):
        super().__init__()
        self.add(TiffMerger())

    def build_batch_context(self, context: Context) -> List[Context]:
        tiff_dir_path = context.get(TIFF_DIR_PATH_KEY)
        merged_dir_path = context.get(MERGED_DIR_PATH_KEY)

        os.makedirs(merged_dir_path, exist_ok=True)

        # Group by date
        file_paths_group_by_date = defaultdict(list)
        for file_path in glob.glob(os.path.join(tiff_dir_path, f"*{TIFF_SUFFIX}")):
            filename = os.path.basename(file_path)
            date = extract_date_from_modis_filename(filename)
            file_paths_group_by_date[date].append(file_path)

        batch_contexts: List[Context] = []
        for date, file_paths in file_paths_group_by_date.items():
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATHS_KEY, file_paths)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(merged_dir_path, f"{date}{TIFF_SUFFIX}"))
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchMultiResampleJob(BatchJob):

    def __init__(self):
        super().__init__()
        self.add(BatchResampleJob())

    def build_batch_context(self, context: Context) -> List[Context]:
        resolution_configs = context.get(RESOLUTION_CONFIGS_KEY)

        batch_contexts = []

        for config in resolution_configs:
            batch_context = context.global_copy()
            batch_context.set(REF_GRID_PATH_KEY, config.ref_grid_path)
            dst_dir_path = os.path.join(context.get(DST_DIR_PATH_KEY), f"{config.resolution_km}km")
            batch_context.set(DST_DIR_PATH_KEY, dst_dir_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchResampleJob(BatchJob):

    def __init__(self):
        super().__init__()
        self.add(TiffResampler())

    def build_batch_context(self, context: Context) -> List[Context]:
        merged_dir_path = context.get(MERGED_DIR_PATH_KEY)
        dst_dir_path = context.get(DST_DIR_PATH_KEY)
        ref_grid_path = context.get(REF_GRID_PATH_KEY)

        os.makedirs(dst_dir_path, exist_ok=True)

        file_paths = glob.glob(os.path.join(merged_dir_path, f"*{TIFF_SUFFIX}"))
        batch_contexts = []

        for file_path in file_paths:
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, file_path)
            batch_context.set(REF_GRID_PATH_KEY, ref_grid_path)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, os.path.basename(file_path)))
            batch_contexts.append(batch_context)

        return batch_contexts


def main():
    job = Job()
    context = Context()

    job.add(BatchConvert2TiffJob())
    job.add(BatchMergeTiffJob())
    job.add(BatchMultiResampleJob())

    # Convert to TIFF Config
    # Read Config
    context.set_global(SRC_DIR_PATH_KEY, INPUT_DIR_PATH)
    # MODIS Data Process Config
    context.set_global(GAP_VALUE_KEY, GAP_VALUE)
    context.set_global(SCALE_FACTOR_KEY, SCALE_FACTOR)
    # Write Config
    context.set_global(TIFF_DIR_PATH_KEY, TIFF_DIR_PATH)

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
    context.set_global(DST_DIR_PATH_KEY, DST_DIR_PATH)

    job.run(context)


if __name__ == "__main__":
    main()
