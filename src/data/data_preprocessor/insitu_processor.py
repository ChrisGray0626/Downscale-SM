#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Handle International Soil Moisture Network (ISMN) data_preprocessor
  @Author Chris
  @Date 2025/6/24
"""
import os.path

import pandas as pd
from ismn.interface import ISMN_Interface
from tqdm import tqdm

from constants import *
from utils.tiff_util import interpolate_tiff
from utils.date_util import is_valid_date

SRC_DIR_PATH = os.path.join(RAW_DIR_PATH, IN_SITU_NAME)
INPUT_FILENAME = "Data_separate_files_header_20160101_20201231_12262_I34f_20251219.zip"
INPUT_PATH = os.path.join(SRC_DIR_PATH, INPUT_FILENAME)
DST_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME)
CSV_PATH = os.path.join(DST_DIR_PATH, f"InSituAggData{CSV_SUFFIX}")
TIFF_DIR_PATH = os.path.join(DST_DIR_PATH, "tiff")


# TODO Refactor to workflow
def extract():
    ismn_data = ISMN_Interface(INPUT_PATH, parallel=True)

    records = []
    for network, station, sensor in ismn_data.collection.iter_sensors(variable='soil_moisture'):
        lon = sensor.metadata['longitude'].val
        lat = sensor.metadata['latitude'].val
        network_name = sensor.metadata['network'].val
        station_name = sensor.metadata['station'].val

        data = sensor.read_data()
        # 清洗数据：质量控制
        data = data[(data['soil_moisture_flag'] == 'G')]
        # 过滤时间 6：00 AM
        data = data[data.index.hour == 6]

        data = data.dropna(subset=['soil_moisture'])

        for row in data.itertuples():
            # 转换日期格式为 YYYYMMDD
            date = pd.to_datetime(row.Index).strftime('%Y%m%d')
            sm = row.soil_moisture
            if sm < 0.02:
                sm = 0.02
            records.append({
                'Network': network_name,
                'Station': station_name,
                'Sensor': str(sensor),
                DATE_NAME: date,
                LONGITUDE_NAME: lon,
                LATITUDE_NAME: lat,
                SM_NAME: sm
            })
    df = pd.DataFrame(records)

    return df


def agg(df: pd.DataFrame):
    result = df.groupby([DATE_NAME, LONGITUDE_NAME, LATITUDE_NAME], as_index=False)[SM_NAME].mean()

    return result


def convert2tiff():
    os.makedirs(TIFF_DIR_PATH, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df_group_by_date = df.groupby(DATE_NAME)
    for date, df in tqdm(df_group_by_date):
        date = str(date)
        if not is_valid_date(date):
            continue
        dst_path = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME, RESOLUTION_36KM, f"{date}{TIFF_SUFFIX}")
        interpolate_tiff(df[SM_NAME].values,
                         df[LONGITUDE_NAME].values,
                         df[LATITUDE_NAME].values,
                         REF_GRID_36KM_PATH,
                         dst_path
                         )


def main():
    os.makedirs(DST_DIR_PATH, exist_ok=True)
    df = extract()
    output_path = os.path.join(DST_DIR_PATH, "InSituSensorData.csv")
    df.to_csv(output_path, index=False)
    df = agg(df)
    output_path = os.path.join(CSV_PATH)
    df.to_csv(output_path, index=False)
    convert2tiff()


if __name__ == '__main__':
    main()
