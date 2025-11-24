#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Job
  @Author Chris
  @Date 2025/11/23
"""
import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from Constant import *
from util.workflow.WorkflowConstant import *
from util.DateUtil import extract_date_from_modis_filename
from util.workflow.Base import *
from util.workflow.Task import *

__all__ = [
    'BatchDecompressJob',
    'BatchMODISData2TiffJob',
    'BatchMergeTiffJob',
    'ResolutionConfig',
    'BatchResampleTiffJob',
    'BatchMultiResampleTiffJob',
    'MultiResampleTiffJob',
]


class BatchDecompressJob(BatchJob):
    def __init__(self, src_dir_path_key: str, dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(Decompressor(
            src_file_path_key=SRC_FILE_PATH_KEY,
            dst_dir_path_key=dst_dir_path_key
        ))

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        batch_contexts = []
        for filename in os.listdir(src_dir_path):
            batch_context = context.copy()
            src_file_path = os.path.join(src_dir_path, filename)
            batch_context.set(SRC_FILE_PATH_KEY, src_file_path)
            batch_context.set(self.dst_dir_path_key, dst_dir_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchMODISData2TiffJob(BatchJob):
    def __init__(self,
                 src_dir_path_key: str,
                 dst_dir_path_key: str,
                 data_name: str,
                 gap_value_key: str,
                 scale_factor_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(
            ValidDateFilter(
                data_name=data_name,
                src_file_path_key=SRC_FILE_PATH_KEY
            ),
            HDF4Reader(src_file_path_key=SRC_FILE_PATH_KEY),
            MODISDataProcessor(
                data_key=DATA_KEY,
                gap_value_key=gap_value_key,
                scale_factor_key=scale_factor_key
            ),
            TiffWriter(
                dst_file_path_key=DST_FILE_PATH_KEY,
                data_key=DATA_KEY,
                transform_key=TRANSFORM_KEY,
                projection_key=PROJECTION_KEY,
                x_size_key=X_SIZE_KEY,
                y_size_key=Y_SIZE_KEY
            ))

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

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
    def __init__(self, src_dir_path_key: str, dst_dir_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(TiffMerger(
            src_file_paths_key=SRC_FILE_PATHS_KEY,
            dst_file_path_key=DST_FILE_PATH_KEY
        ))

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

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


@dataclass
class ResolutionConfig:
    resolution_km: int
    ref_grid_path: str


class BatchMultiResampleTiffJob(BatchJob):
    def __init__(self,
                 resolution_configs_key: str,
                 src_dir_path_key: str,
                 ref_grid_path_key: str,
                 dst_dir_path_key: str,
                 ):
        super().__init__()
        self.resolution_configs_key = resolution_configs_key
        self.src_dir_path_key = src_dir_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(BatchResampleTiffJob(
            src_dir_path_key=src_dir_path_key,
            ref_grid_path_key=ref_grid_path_key,
            dst_dir_path_key=dst_dir_path_key,
        ))

    def build_batch_context(self, context: Context) -> List[Context]:
        resolution_configs = context.get(self.resolution_configs_key)

        batch_contexts = []

        for config in resolution_configs:
            batch_context = context.global_copy()
            dst_dir_path = os.path.join(context.get(self.dst_dir_path_key), f"{config.resolution_km}km")
            batch_context.set(self.dst_dir_path_key, dst_dir_path)
            batch_context.set(self.ref_grid_path_key, config.ref_grid_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class MultiResampleTiffJob(BatchJob):

    def __init__(self,
                 resolution_configs_key: str,
                 src_file_path_key: str,
                 ref_grid_path_key: str,
                 dst_dir_path_key: str,
                 ):
        super().__init__()
        self.resolution_configs_key = resolution_configs_key
        self.src_file_path_key = src_file_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.add(TiffResampler(
                src_file_path_key=src_file_path_key,
                ref_grid_path_key=ref_grid_path_key,
                dst_file_path_key=DST_FILE_PATH_KEY
            ))

    def build_batch_context(self, context: Context) -> List[Context]:
        resolution_configs = context.get(self.resolution_configs_key)
        src_file_path = context.get(self.src_file_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []
        for config in resolution_configs:
            batch_context = context.global_copy()
            batch_context.set(self.src_file_path_key, src_file_path)
            batch_context.set(self.ref_grid_path_key, config.ref_grid_path)
            dst_file_path = os.path.join(dst_dir_path, f"{config.resolution_km}km", os.path.basename(src_file_path))
            batch_context.set(DST_FILE_PATH_KEY, dst_file_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class BatchResampleTiffJob(BatchJob):
    def __init__(self,
                 src_dir_path_key: str,
                 dst_dir_path_key: str,
                 ref_grid_path_key: str):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.add(TiffResampler(
            src_file_path_key=SRC_FILE_PATH_KEY,
            ref_grid_path_key=REF_GRID_PATH_KEY,
            dst_file_path_key=DST_FILE_PATH_KEY
        ))

    def build_batch_context(self, context: Context) -> List[Context]:
        src_dir_path = context.get(self.src_dir_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)
        ref_grid_path = context.get(self.ref_grid_path_key)

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



