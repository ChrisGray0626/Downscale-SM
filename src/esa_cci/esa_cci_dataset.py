#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/27
"""
import os

from constants import DATA_DIR_PATH, ESA_CCI_NAME
from datasets.base_data_store import BaseTiffStore


class ESACCIStore(BaseTiffStore):
    def __init__(self, resolution: str):
        base_dir = os.path.join(DATA_DIR_PATH, ESA_CCI_NAME)
        super().__init__(base_dir=base_dir, resolution=resolution)
