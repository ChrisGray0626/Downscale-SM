#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Merge Task & Job
  @Author Chris
  @Date 2025/11/25
"""
import glob
import os
from collections import defaultdict
from typing import List

from Constant import TIFF_SUFFIX
from util.DateUtil import extract_date_from_modis_filename
from util.TiffUtil import merge_tiff
from util.workflow.WorkflowConstant import SRC_FILE_PATHS_KEY, DST_FILE_PATH_KEY
from util.workflow.core.Base import BaseTask, Context, BatchJob

__all__ = [
    'BatchMergeTiffJob',
    'TiffMerger',
]


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


class TiffMerger(BaseTask):
    def __init__(self,
                 dst_file_path_key: str,
                 src_file_paths_key: str = None,
                 src_dir_path_key: str = None):
        super().__init__()
        if src_file_paths_key is None and src_dir_path_key is None:
            raise ValueError("Either src_file_paths_key or src_dir_path_key must be provided.")
        self.src_file_paths_key = src_file_paths_key
        self.src_dir_path_key = src_dir_path_key
        self.dst_file_path_key = dst_file_path_key

    def execute(self, context) -> Context:
        dst_file_path = context.get(self.dst_file_path_key)
        src_file_paths = context.get(self.src_file_paths_key, None)
        src_dir_path = context.get(self.src_dir_path_key, None)

        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

        merge_tiff(
            dst_file_path=dst_file_path,
            src_file_paths=src_file_paths,
            src_dir_path=src_dir_path
        )

        return context
