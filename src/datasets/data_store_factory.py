#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2026/1/27
"""
from constants import (
    ESA_CCI_NAME,
    RF_NAME,
    DDPM_IMAGE_NAME,
    GWR_NAME,
    RESNET_NAME,
    SM_NAME,
    IN_SITU_NAME,
    DDPM_PIXEL_NAME,
)
from datasets.common_data_store import SMAPStore, InsituStore
from ddpm_image.image_dataset import (
    DDPMImageCorrectionResultStore,
    DDPMImageInferenceResultStore,
)
from ddpm_pixel.pixel_dataset import (
    DDPMPixelCorrectionResultStore,
    DDPMPixelInferenceResultStore,
)
from esa_cci.esa_cci_dataset import ESACCIStore
from gwr.gwr_dataset import GWRResultStore
from resnet.resnet_dataset import ResNetResultStore
from rf.rf_dataset import RFResultStore

CONTRACT_RESULT_STORE_REGISTRY = {
    ESA_CCI_NAME: ESACCIStore,
    RF_NAME: RFResultStore,
    GWR_NAME: GWRResultStore,
    RESNET_NAME: ResNetResultStore,
    SM_NAME: SMAPStore,
    IN_SITU_NAME: InsituStore,
}

DDPM_RESULT_STORE_REGISTRY = {
    DDPM_IMAGE_NAME: {
        False: DDPMImageInferenceResultStore,
        True: DDPMImageCorrectionResultStore,
    },
    DDPM_PIXEL_NAME: {
        False: DDPMPixelInferenceResultStore,
        True: DDPMPixelCorrectionResultStore,
    },
}


def build(product_name, resolution, is_correction: bool = False):
    if product_name in DDPM_RESULT_STORE_REGISTRY:
        cls = DDPM_RESULT_STORE_REGISTRY[product_name][bool(is_correction)]
    elif product_name in CONTRACT_RESULT_STORE_REGISTRY:
        cls = CONTRACT_RESULT_STORE_REGISTRY[product_name]
    else:
        raise ValueError(f"Unknown product name: {product_name}")
    return cls(resolution=resolution)
