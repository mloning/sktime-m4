#!/usr/bin/env python3 -u

__author__ = ["Markus Löning"]

import os
import socket
import time
from warnings import filterwarnings
from warnings import simplefilter

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from sklearn.base import clone
from sktime.exceptions import FitFailedWarning
from sktime.performance_metrics.forecasting import mase_loss
from sktime.performance_metrics.forecasting import smape_loss

from config import DATADIR
from config import DATASETS
from config import N_JOBS
from config import REPLICATED_RESULTS_DIR
from config import SELECTED_FORECASTERS
from config import TESTDIR
from config import TRAINDIR
from construct import construct_forecasters
from utils import load_metadata
from utils import save_results

# suppress TensorFlow info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# we limit number of threads we use, parallelising over
# CPUs sometimes led to crashes for some models
os.environ['OMP_NUM_THREADS'] = '1'


def _fit_evaluate(i, forecaster, y_train, y_test, fh, sp):
    """Fit and evaluate single forecaster on single series"""
    # silence warnings
    filterwarnings("ignore", module="sklearn")
    filterwarnings("ignore", module="statsmodels")
    simplefilter("ignore", category=FitFailedWarning)
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=UserWarning)

    # remove padding
    y_train = y_train[~np.isnan(y_train)]
    y_test = y_test[~np.isnan(y_test)]

    # check forecasting horizon
    assert len(fh) == len(y_test)

    # get data into expected format
    y_train = pd.Series(y_train)
    n_timepoints = len(y_train)

    # adjust test index to be after train index
    index = np.arange(n_timepoints, n_timepoints + len(y_test))
    y_test = pd.Series(y_test, index=index)
    assert y_test.index[0] == y_train.index[-1] + 1

    # clone forecaster
    f = clone(forecaster)

    # fit
    start = time.time()
    f.fit(y_train, fh=fh)
    fit_time = time.time() - start

    # predict
    start = time.time()
    y_pred = f.predict()
    predict_time = time.time() - start

    # compute errors
    mase = mase_loss(y_test, y_pred, y_train, sp=sp)
    smape = smape_loss(y_test, y_pred)

    # collect and return results
    results = {
        "id": i,
        "y_pred": y_pred,
        "mase": mase,
        "smape": smape,
        "fit_time": fit_time,
        "predict_time": predict_time
    }
    return results


def main():
    """Fit and evaluate selected forecasters on M4 data set"""
    print("Host: ", socket.gethostname())
    print("Results directory: ", REPLICATED_RESULTS_DIR)
    print("Selected forecasters: ", SELECTED_FORECASTERS)
    print("Selected datasets: ", DATASETS)

    # import meta data
    meta = load_metadata(DATADIR)

    # dictionary of forecasting horizons and seasonal periodicities
    FHS = meta.set_index("SP")["Horizon"].to_dict()
    SPS = meta.set_index("SP")["Frequency"].to_dict()

    # iterate over M4 data sets (grouped by sampling frequency)
    for dataset in DATASETS:
        print(f"Dataset: {dataset} ...")  # print status

        # get forecasting horizon
        fh = np.arange(FHS[dataset]) + 1

        # get seasonal frequency
        sp = SPS[dataset]

        # define and select models
        FORECASTERS = construct_forecasters(sp=sp, fh=fh)
        FORECASTERS = {name: forecaster for name, forecaster in
                       FORECASTERS.items() if name in SELECTED_FORECASTERS}

        # load train and test data
        alltrain = pd.read_csv(os.path.join(TRAINDIR, f"{dataset}-train.csv"),
                               index_col=0)
        alltest = pd.read_csv(os.path.join(TESTDIR, f"{dataset}-test.csv"),
                              index_col=0)

        # ensure correct ordering of series, and use numpy arrays
        # for more efficient parallelization with shared memory
        alltrain = alltrain.sort_index()
        series_ids = alltrain.index
        alltrain = alltrain.reset_index(drop=True).values
        alltest = alltest.sort_index().reset_index(drop=True).values

        # check number of series in dataset
        n_series = alltrain.shape[0]
        assert n_series == meta.loc[:, "SP"].value_counts()[dataset]

        # iterate over forecasting models
        for name, forecaster in FORECASTERS.items():
            print(f"{name} ...")

            # create model directory if necessary
            filedir = os.path.join(REPLICATED_RESULTS_DIR, name)
            if not os.path.isdir(filedir):
                os.makedirs(filedir)

            # if results file already exists, skip series
            filename = os.path.join(filedir, f"{name}_{dataset}")
            if os.path.isfile(f"{filename}_y_pred.txt"):
                print(f"{name} skipped, forecasts already exist.")
                continue

            # catch errors to continue with remaining models, even when some
            # models fail
            try:
                # iterate over individual series in data set
                results = Parallel(n_jobs=N_JOBS)(
                    delayed(_fit_evaluate)(
                        series_id,
                        forecaster,
                        alltrain[i, :],
                        alltest[i, :],
                        fh,
                        sp
                    )
                    for i, series_id in enumerate(series_ids)
                )

                # save results
                save_results(filename, results)

            # raise key exceptions
            except (TerminatedWorkerError, KeyboardInterrupt, SystemExit):
                raise

            # catch all other exceptions and continue
            except Exception as e:
                print(
                    f"{name} skipped, error during fitting or forecasting: "
                    f"{e}")
                continue

    print("Done.")


if __name__ == "__main__":
    main()
