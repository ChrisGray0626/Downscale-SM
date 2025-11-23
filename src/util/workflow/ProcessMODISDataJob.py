#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/23
"""
import glob
from collections import defaultdict

from constant import *
from util.date_util import extract_date_from_modis_filename
from util.workflow.Task import *
from dataclasses import dataclass


@dataclass
class ResolutionConfig:
    resolution_km: int
    ref_grid_path: str


class BatchConvert2TiffJob(BatchJob):

    required_context_keys = (RAW_DIR_PATH_KEY, CONVERTED_DIR_PATH_KEY)

    def __init__(self):
        super().__init__()
        self.add(
            HDF4Reader(),
            MODISDataProcessor(),
            TiffWriter())

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(RAW_DIR_PATH_KEY)
        dst_dir_path = context.get(CONVERTED_DIR_PATH_KEY)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []
        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{HDF4_SUFFIX}"))
        for src_file_path in src_file_paths:
            filename = os.path.basename(src_file_path)
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)

            dst_path = os.path.join(dst_dir_path, filename.replace(HDF4_SUFFIX, TIFF_SUFFIX))
            batch_context.set(DST_FILE_PATH_KEY, dst_path)

            batch_contexts.append(batch_context)

        return batch_contexts


class BatchMergeTiffJob(BatchJob):

    required_context_keys = (CONVERTED_DIR_PATH_KEY, MERGED_DIR_PATH_KEY)

    def __init__(self):
        super().__init__()
        self.add(TiffMerger())

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(CONVERTED_DIR_PATH_KEY)
        dst_dir_path = context.get(MERGED_DIR_PATH_KEY)

        os.makedirs(dst_dir_path, exist_ok=True)

        # Group by date
        src_file_paths_group_by_date = defaultdict(list)
        for src_file_path in glob.glob(os.path.join(src_dir_path, f"*{TIFF_SUFFIX}")):
            filename = os.path.basename(src_file_path)
            date = extract_date_from_modis_filename(filename)
            src_file_paths_group_by_date[date].append(src_file_path)

        batch_contexts: List[Context] = []
        for date, src_file_paths in src_file_paths_group_by_date.items():
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATHS_KEY, src_file_paths)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}"))
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchMultiResampleTiffJob(BatchJob):

    required_context_keys = (RESOLUTION_CONFIGS_KEY)

    def __init__(self):
        super().__init__()
        self.add(BatchResampleTiffJob())

    def build_batch_context(self, context: Context) -> List[Context]:
        resolution_configs = context.get(RESOLUTION_CONFIGS_KEY)

        batch_contexts = []

        for config in resolution_configs:
            batch_context = context.global_copy()
            batch_context.set(REF_GRID_PATH_KEY, config.ref_grid_path)
            resampled_dir_path = os.path.join(context.get(RESAMPLED_DIR_PATH_KEY), f"{config.resolution_km}km")
            batch_context.set(RESAMPLED_DIR_PATH_KEY, resampled_dir_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchResampleTiffJob(BatchJob):

    required_context_keys = (MERGED_DIR_PATH_KEY, RESAMPLED_DIR_PATH_KEY, REF_GRID_PATH_KEY)

    def __init__(self):
        super().__init__()
        self.add(TiffResampler())

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(MERGED_DIR_PATH_KEY)
        dst_dir_path = context.get(RESAMPLED_DIR_PATH_KEY)
        ref_grid_path = context.get(REF_GRID_PATH_KEY)

        os.makedirs(dst_dir_path, exist_ok=True)

        src_file_paths = glob.glob(os.path.join(src_dir_path, f"*{TIFF_SUFFIX}"))
        batch_contexts = []

        for src_file_path in src_file_paths:
            batch_context = context.global_copy()
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)
            batch_context.set(REF_GRID_PATH_KEY, ref_grid_path)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, os.path.basename(src_file_path)))
            batch_contexts.append(batch_context)

        return batch_contexts

