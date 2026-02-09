#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/23
"""
import glob
import os
import re
from datetime import timedelta, datetime

from constants import NDVI_NAME, RESOLUTION_1KM, TIFF_SUFFIX, VALID_DATE_FILE_PATH, PROCESSED_DIR_PATH
from utils.util import write_txt, read_txt


def extract_date_from_modis_filename(filename: str):
    pattern = re.compile(r"^\w*?\.A(\d{7})\.h\d{2}v\d{2}\..*\.\w{3}$")
    match = pattern.match(filename)
    year_day = match.group(1)
    date = convert_year_day2date(year_day)

    return date


def convert_year_day2date(year_day):
    # 解析年份和年积日
    year = int(year_day[:4])
    day_of_year = int(year_day[4:])

    # 构造当年的1月1日，然后加上年积日偏移
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    # 转换为 YYYYMMDD 格式
    return date.strftime('%Y%m%d')


def handle_valid_date():
    dir_path = os.path.join(PROCESSED_DIR_PATH, NDVI_NAME, RESOLUTION_1KM)
    filenames = os.listdir(dir_path)
    # remove the suffix
    dates = [os.path.splitext(f)[0] for f in filenames if f.endswith(TIFF_SUFFIX)]
    dates = sorted(dates)

    write_txt(VALID_DATE_FILE_PATH, dates)


def is_valid_date(date: str):
    tgt_dates = read_txt(VALID_DATE_FILE_PATH)
    return date in tgt_dates


def get_valid_dates():
    tgt_dates = read_txt(VALID_DATE_FILE_PATH)
    return tgt_dates


def list_date_from_dir(dir_path: str, suffix: str = TIFF_SUFFIX):
    if not os.path.isdir(dir_path):
        return []
    out = []
    for p in glob.glob(os.path.join(dir_path, f"*{suffix}")):
        m = re.search(r"(\d{8})", os.path.basename(p))
        if m:
            out.append(m.group(1))
    return sorted(set(out))
