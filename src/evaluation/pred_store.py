#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/27
"""
from constants import ESA_CCI_NAME, RF_NAME, DDPM_NAME
from datasets.dataset import CorrectionResultStore, InferenceResultStore
from esa_cci.esa_cci_dataset import ESACCIStore
from rf.rf_dataset import RFResultStore


def build_pred_store(product_name, resolution, is_correction=False):
    if product_name == ESA_CCI_NAME:
        return ESACCIStore(resolution=resolution)
    if product_name == RF_NAME:
        return RFResultStore(resolution=resolution)
    if product_name == DDPM_NAME:
        if is_correction:
            return CorrectionResultStore(resolution=resolution)
        else:
            return InferenceResultStore(resolution=resolution)
    raise ValueError(f"Unknown product name: {product_name}")
