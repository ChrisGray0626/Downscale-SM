#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Runner entry point for regenerating paper-ready evaluation outputs
  @Author Chris
  @Date 2026/3/9
"""

from constants import RESOLUTION_1KM, DDPM_PIXEL_NAME, RESOLUTION_36KM
from ddpm_pixel import pixel_inferencer
from evaluation import heterogeneity_stratified_evaluator
from evaluation import insitu_grid_evaluator
from evaluation import insitu_timeseries_evaluator
from evaluation import paper_summary_builder
from evaluation import upscale_evaluator


def main():
    upscale_evaluator.PROD_NAMES = [DDPM_PIXEL_NAME]
    upscale_evaluator.main()
    insitu_grid_evaluator.PROD_RESOLUTIONS = {
        DDPM_PIXEL_NAME: [RESOLUTION_36KM],
    }
    insitu_grid_evaluator.main()
    pixel_inferencer.RESOLUTION = RESOLUTION_1KM
    pixel_inferencer.main()


def _evaluate():
    heterogeneity_stratified_evaluator.main()
    insitu_grid_evaluator.main()
    upscale_evaluator.main()
    paper_summary_builder.main()
    insitu_timeseries_evaluator.main()


if __name__ == "__main__":
    main()
