#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Handle International Soil Moisture Network (ISMN) data
@Author Chris
@Date 2025/6/24
"""
import os
from typing import List

import pandas as pd
from ismn.interface import ISMN_Interface

from constants import *
from utils.raster_util import interpolate_tiff
from utils.workflow.core.base import BaseTask, Context, BatchJob, Job
from utils.workflow.core.context_key import *
from utils.workflow.common.Resampler import ResolutionConfig

SRC_DIR_PATH = os.path.join(RAW_DIR_PATH, IN_SITU_NAME)
SRC_FILENAME = "Data_separate_files_header_20160101_20201231_12262_I34f_20251219.zip"
SRC_FILE_PATH = os.path.join(SRC_DIR_PATH, SRC_FILENAME)
DST_DIR_PATH = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME)


def main():
    job = Job()
    context = Context()

    job.add([
        ISMNExtractor(),
        BatchMultiInterpolationJob()
    ])

    context.set_global(SRC_FILE_PATH_KEY, SRC_FILE_PATH)
    context.set_global(DST_DIR_PATH_KEY, DST_DIR_PATH)
    context.set_global(RESOLUTION_CONFIGS_KEY, [
        ResolutionConfig(
            resolution_km=1,
            ref_grid_path=REF_GRID_1KM_PATH,
        ),
        ResolutionConfig(
            resolution_km=36,
            ref_grid_path=REF_GRID_36KM_PATH,
        ),
    ])

    job.run(context)


class ISMNExtractor(BaseTask):

    def __init__(self, src_file_path_key: str = SRC_FILE_PATH_KEY, data_key: str = DATA_KEY):
        super().__init__()
        self.src_file_path_key = src_file_path_key
        self.data_key = data_key

    def execute(self, context: Context) -> Context:
        src_file_path = context.get(self.src_file_path_key)
        ismn_data = ISMN_Interface(src_file_path, parallel=True)

        records = []
        for network, station, sensor in ismn_data.collection.iter_sensors(variable='soil_moisture'):
            lon = sensor.metadata['longitude'].val
            lat = sensor.metadata['latitude'].val

            data = sensor.read_data()
            data = data[(data['soil_moisture_flag'] == 'G')]
            data = data[data.index.hour == 6]
            data = data.dropna(subset=['soil_moisture'])

            for idx, row in data.iterrows():
                date = idx.strftime('%Y%m%d')
                sm = float(row['soil_moisture'])
                if sm < 0.02:
                    sm = 0.02
                records.append({
                    DATE_NAME: date,
                    LONGITUDE_NAME: lon,
                    LATITUDE_NAME: lat,
                    SM_NAME: sm
                })

        df = pd.DataFrame(records)
        aggregated = df.groupby([DATE_NAME, LONGITUDE_NAME, LATITUDE_NAME], as_index=False)[SM_NAME].mean()

        context.set_global(self.data_key, aggregated)
        return context


class BatchMultiInterpolationJob(BatchJob):

    def __init__(self,
                 data_key: str = DATA_KEY,
                 dst_dir_path_key: str = DST_DIR_PATH_KEY,
                 resolution_configs_key: str = RESOLUTION_CONFIGS_KEY):
        super().__init__()
        self.data_key = data_key
        self.dst_dir_path_key = dst_dir_path_key
        self.resolution_configs_key = resolution_configs_key
        self.add(BatchInterpolationJob(
            data_key=data_key,
            dst_dir_path_key=dst_dir_path_key,
            ref_grid_path_key=REF_GRID_PATH_KEY
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


class BatchInterpolationJob(BatchJob):

    def __init__(self,
                 data_key: str = DATA_KEY,
                 dst_dir_path_key: str = DST_DIR_PATH_KEY,
                 ref_grid_path_key: str = REF_GRID_PATH_KEY):
        super().__init__()
        self.data_key = data_key
        self.dst_dir_path_key = dst_dir_path_key
        self.ref_grid_path_key = ref_grid_path_key
        self.add([
            InsituInterpolator()
        ])

    def build_batch_context(self, context: Context) -> List[Context]:
        aggregated_df = context.get(self.data_key)
        dst_dir_path = context.get(self.dst_dir_path_key)
        ref_grid_path = context.get(self.ref_grid_path_key)

        os.makedirs(dst_dir_path, exist_ok=True)

        batch_contexts = []
        for date, group_df in aggregated_df.groupby(DATE_NAME):
            batch_context = context.global_copy()
            batch_context.set(DATE_KEY, date)
            batch_context.set(DATA_KEY, group_df)
            batch_context.set(self.ref_grid_path_key, ref_grid_path)
            batch_context.set(DST_DIR_PATH_KEY, dst_dir_path)
            batch_context.set(DST_FILE_PATH_KEY, os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}"))
            batch_contexts.append(batch_context)

        return batch_contexts


class InsituInterpolator(BaseTask):

    def __init__(self,
                 data_key: str = DATA_KEY,
                 ref_grid_path_key: str = REF_GRID_PATH_KEY,
                 dst_file_path_key: str = DST_FILE_PATH_KEY):
        super().__init__()
        self.data_key = data_key
        self.ref_grid_path_key = ref_grid_path_key
        self.dst_file_path_key = dst_file_path_key

    def execute(self, context: Context) -> Context:
        df = context.get(self.data_key)
        ref_grid_path = context.get(self.ref_grid_path_key)
        dst_file_path = context.get(self.dst_file_path_key)

        interpolate_tiff(
            df[SM_NAME].values,
            df[LONGITUDE_NAME].values,
            df[LATITUDE_NAME].values,
            ref_grid_path,
            dst_file_path
        )

        return context


if __name__ == '__main__':
    main()
