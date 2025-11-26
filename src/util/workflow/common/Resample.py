#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Resample Task & Job
  @Author Chris
  @Date 2025/11/25
"""
import glob
import os
from dataclasses import dataclass
from typing import List

from Constant import TIFF_SUFFIX
from util.TiffUtil import resample_tiff
from util.workflow.core.ContextKey import SRC_FILE_PATH_KEY, DST_FILE_PATH_KEY, REF_GRID_PATH_KEY, \
    RESOLUTION_CONFIGS_KEY
from util.workflow.core.Base import Context, BaseTask, BatchJob

__all__ = [
    'BatchMultiResampleTiffJob',
    'MultiResampleTiffJob',
    'BatchResampleTiffJob',
    'TiffResampler',
    'ResolutionConfig',
]


class BatchMultiResampleTiffJob(BatchJob):

    def __init__(self,
                 src_dir_path_key: str,
                 dst_dir_path_key: str,
                 resolution_configs_key: str = RESOLUTION_CONFIGS_KEY,
                 ):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.resolution_configs_key = resolution_configs_key
        self.add(BatchResampleTiffJob(
            src_dir_path_key=src_dir_path_key,
            dst_dir_path_key=dst_dir_path_key,
            ref_grid_path_key=REF_GRID_PATH_KEY,
        ))

    def build_batch_context(self, context: Context) -> List[Context]:
        resolution_configs = context.get(self.resolution_configs_key)

        batch_contexts = []

        for config in resolution_configs:
            batch_context = context.global_copy()
            dst_dir_path = os.path.join(context.get(self.dst_dir_path_key), f"{config.resolution_km}km")
            batch_context.set(self.dst_dir_path_key, dst_dir_path)
            batch_context.set(REF_GRID_PATH_KEY, config.ref_grid_path)
            batch_contexts.append(batch_context)

        return batch_contexts


class MultiResampleTiffJob(BatchJob):

    def __init__(self,
                 src_file_path_key: str,
                 ref_grid_path_key: str,
                 dst_dir_path_key: str,
                 resolution_configs_key: str = RESOLUTION_CONFIGS_KEY,
                 ):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.resolution_configs_key = resolution_configs_key
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
                 ref_grid_path_key: str = REF_GRID_PATH_KEY,
                 ):
        super().__init__()
        self.src_dir_path_key = src_dir_path_key
        self.dst_dir_path_key = dst_dir_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.add(TiffResampler(
            src_file_path_key=SRC_FILE_PATH_KEY,
            ref_grid_path_key=ref_grid_path_key,
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
            batch_context.set(self.ref_grid_path_key, ref_grid_path)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, os.path.basename(src_file_path)))
            batch_contexts.append(batch_context)

        return batch_contexts


class TiffResampler(BaseTask):
    def __init__(self,
                 src_file_path_key: str = SRC_FILE_PATH_KEY,
                 ref_grid_path_key: str = REF_GRID_PATH_KEY,
                 dst_file_path_key: str = DST_FILE_PATH_KEY,
                 ):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_file_path_key = dst_file_path_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        ref_grid_path = context.get(self.ref_grid_path_key)
        dst_path = context.get(self.dst_file_path_key)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        resample_tiff(src_path, ref_grid_path, dst_path)

        return context


@dataclass
class ResolutionConfig:
    resolution_km: int
    ref_grid_path: str
