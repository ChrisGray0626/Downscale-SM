#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/27
"""
from constants import ESA_CCI_NAME, RF_NAME, DDPM_NAME, GWR_NAME, RESNET_NAME, SM_NAME, IN_SITU_NAME, PIXEL_DDPM_NAME
from datasets.dataset import CorrectionResultStore, InferenceResultStore, SMAPStore, InsituStore, PixelInferenceStore
from esa_cci.esa_cci_dataset import ESACCIStore
from gwr.gwr_dataset import GWRResultStore
from resnet.resnet_dataset import ResNetResultStore
from rf.rf_dataset import RFResultStore


def build_pred_store(product_name, resolution, is_correction=False):
    if product_name == ESA_CCI_NAME:
        return ESACCIStore(resolution=resolution)
    elif product_name == RF_NAME:
        return RFResultStore(resolution=resolution)
    elif product_name == GWR_NAME:
        return GWRResultStore(resolution=resolution)
    elif product_name == RESNET_NAME:
        return ResNetResultStore(resolution=resolution)
    elif product_name == SM_NAME:
        return SMAPStore(resolution=resolution)
    elif product_name == IN_SITU_NAME:
        return InsituStore(resolution=resolution)
    elif product_name == DDPM_NAME:
        if is_correction:
            return CorrectionResultStore(resolution=resolution)
        else:
            return InferenceResultStore(resolution=resolution)
    elif product_name == PIXEL_DDPM_NAME:
        return PixelInferenceStore(resolution=resolution)
    raise ValueError(f"Unknown product name: {product_name}")
