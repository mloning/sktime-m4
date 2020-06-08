#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

if __name__ == "__main__":
    from config import DATASETS, M4_FORECASTERS
    from config import REPLICATED_RESULTS_DIR, ORIGINAL_RESULTS_DIR
    from utils import compute_and_save_mean_metrics
    from utils import compute_and_save_difference_in_metrics, find_forecasters

    # replicated results
    print("Evaluating replicated results ...")
    FORECASTERS = find_forecasters(REPLICATED_RESULTS_DIR, DATASETS)
    print(f"Forecasters: {FORECASTERS}")
    compute_and_save_mean_metrics(REPLICATED_RESULTS_DIR, DATASETS,
                                  FORECASTERS)

    # compute metrics from original results
    print("Evaluating original results ...")
    FORECASTERS = find_forecasters(ORIGINAL_RESULTS_DIR, DATASETS)
    print(f"Forecasters: {FORECASTERS}")
    compute_and_save_mean_metrics(ORIGINAL_RESULTS_DIR, DATASETS, FORECASTERS)

    # compute difference in results
    print("Comparing results ...")
    compute_and_save_difference_in_metrics(REPLICATED_RESULTS_DIR,
                                           ORIGINAL_RESULTS_DIR, DATASETS,
                                           M4_FORECASTERS)

    print("Done.")
