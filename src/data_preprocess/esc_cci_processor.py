#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/27
"""
import os

from constants import ESA_CCI_NAME, DATA_DIR_PATH, RAW_NAME, PROCESSED_NAME

DATA_NAME = ESA_CCI_NAME
ROOT_DIR_PATH = os.path.join(DATA_DIR_PATH, DATA_NAME)
RAW_DIR_PATH = os.path.join(ROOT_DIR_PATH, RAW_NAME)
PROCESSED_DIR_PATH = os.path.join(ROOT_DIR_PATH, PROCESSED_NAME)

