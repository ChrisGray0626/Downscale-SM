#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Write Task
  @Author Chris
  @Date 2025/11/26
"""
import os

import numpy as np

from util.TiffUtil import write_tiff_from_transform
from util.workflow.core.Base import BaseTask
from util.workflow.core.ContextKey import DATA_KEY, TRANSFORM_KEY, EPSG_CODE_KEY, CRS_KEY, DST_FILE_PATH_KEY

__all__ = [
    "TiffWriter",
]


class TiffWriter(BaseTask):
    def __init__(self,
                 dst_file_path_key: str = DST_FILE_PATH_KEY,
                 data_key: str = DATA_KEY,
                 transform_key: str = TRANSFORM_KEY,
                 epsg_code_key: str = EPSG_CODE_KEY,
                 crs_key: str = CRS_KEY,
                 ):
        super().__init__()
        self.dst_file_path_key = dst_file_path_key
        self.data_key = data_key
        self.transform_key = transform_key
        self.epsg_code_key = epsg_code_key
        self.crs_key = crs_key

    def execute(self, context):
        dst_file_path = context.get(self.dst_file_path_key)
        data = context.get(self.data_key)
        transform = context.get(self.transform_key)
        epsg_code = context.get(self.epsg_code_key)
        crs = context.get(self.crs_key)

        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

        write_tiff_from_transform(
            data=data,
            dst_file_path=dst_file_path,
            transform=transform,
            epsg_code=epsg_code,
            crs=crs,
            nodata=np.nan,
            dtype=np.float32,
        )

        return context
