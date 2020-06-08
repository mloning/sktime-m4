#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np

from config import DATASETS
from config import M4_FORECASTERS
from config import ORIGINAL_RESULTS_DIR
from config import REFERENCE_FORECASTERS
from config import REPODIR
from utils import compute_and_save_mean_metrics
from utils import compute_and_save_original_metrics
from utils import load_published_mean_metrics

# there are two sources of original results, a table of aggregated metrics
# provided
# in the repo and published in the paper on the M4 study and and individual
# forecasted values

# select forecasters
forecasters = list(M4_FORECASTERS)
forecasters.extend(REFERENCE_FORECASTERS)

# we first compute score from individual forecasts, but then check them
# against the aggregated results
print("Preparing original results ...")
compute_and_save_original_metrics(REPODIR, ORIGINAL_RESULTS_DIR, forecasters)
smape_orig, mase_orig, owa_orig = compute_and_save_mean_metrics(
    ORIGINAL_RESULTS_DIR, DATASETS, forecasters)

# check against published results
print("Checking original results against published results ...")
smape_pub, mase_pub, owa_pub = load_published_mean_metrics(REPODIR, DATASETS,
                                                           forecasters)

metrics_orig = {
    "smape": smape_orig,
    "mase": mase_orig,
    "owa": owa_orig
}
metrics_pub = {
    "smape": smape_pub,
    "mase": mase_pub,
    "owa": owa_pub,
}

for metric in ["smape", "mase", "owa"]:
    print(f"Checking {metric} ...")

    # drop ARIMA which for some reason is not in the table of aggregated
    # results, re-order columns
    orig = metrics_orig[metric].drop(index="ARIMA").loc[:, DATASETS]
    publ = metrics_pub[metric].loc[:, DATASETS]

    np.testing.assert_array_almost_equal(orig.values, publ.values, decimal=3)

print("Done.")
