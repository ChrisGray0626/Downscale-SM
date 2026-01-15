#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Decompress Task & Job
  @Author Chris
  @Date 2025/11/25
"""
import os
from typing import List

from constants import ZIP_SUFFIX
from utils.util import unzip_file
from utils.workflow.core.base import BaseTask, Context, BatchJob
from utils.workflow.core.context_key import SRC_FILE_PATH_KEY, DST_DIR_PATH_KEY

__all__ = [
    'BatchDecompressJob',
    'Decompressor',
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


class Decompressor(BaseTask):
    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY,
                 dst_dir_path_key: str = DST_DIR_PATH_KEY
                 ):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.dst_dir_path_key = dst_dir_path_key

    def execute(self, context) -> Context:
        src_path = context.get(self.src_file_path_key)
        dst_dir_path = context.get(self.dst_dir_path_key)

        if src_path.endswith(ZIP_SUFFIX):
            unzip_file(src_path, dst_dir_path)

        return context
