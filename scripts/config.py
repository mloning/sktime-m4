#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import os

# list of replicated forecasters from
# the M4 study
M4_FORECASTERS = [
    "Naive",
    "Naive2",
    "sNaive",
    "SES",
    "Holt",
    "Damped",
    "Com",
    "ARIMA",
    "Theta",
    "260",  # theta with box-cox
    "MLP",
    "RNN",
    # "ETS"
]

# submission numbers of forecasters used
# for comparison
REFERENCE_FORECASTERS = [
    "118",
    "211",
    # "260",
    "245",
]
REFERENCE_FORECASTERS_LOOKUP = {
    "118": "M4 winner",  # winner
    "245": "M4 runner-up",  # runner-up
    "211": "M4 best pure ML",  # best pure ML (Tratto)
    "260": "Theta-bc"  # best pure stats
}

# select forecasters to evaluate
SELECTED_FORECASTERS = M4_FORECASTERS

# M4 data set and performance metric names
DATASETS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
METRICS = ["smape", "mase", "owa"]

# paths
HOME = os.path.expanduser("~")
REPODIR = os.path.join(HOME, "Documents/Research/papers/m4-replication/")
DATADIR = os.path.join(REPODIR, "Dataset")
TRAINDIR = os.path.join(DATADIR, "Train")
TESTDIR = os.path.join(DATADIR, "Test")
REPLICATED_RESULTS_DIR = os.path.join(REPODIR, "results/replicated")
ORIGINAL_RESULTS_DIR = os.path.join(REPODIR, "results/original")
TABLES_DIR = os.path.join(
    HOME, "Documents/Research/papers/sktime_forecasting_m4_replication/tables")

assert os.path.exists(DATADIR)
assert os.path.exists(TRAINDIR)
assert os.path.exists(TESTDIR)

if not os.path.exists(ORIGINAL_RESULTS_DIR):
    print(f"Creating results folder: {ORIGINAL_RESULTS_DIR}")
    os.makedirs(ORIGINAL_RESULTS_DIR)

if not os.path.exists(REPLICATED_RESULTS_DIR):
    print(f"Creating results folder: {REPLICATED_RESULTS_DIR}")
    os.makedirs(REPLICATED_RESULTS_DIR)

if not os.path.exists(TABLES_DIR):
    print(f"Creating tables folder: {TABLES_DIR}")
    os.makedirs(TABLES_DIR)

# number of CPUs to use
N_JOBS = os.cpu_count()
