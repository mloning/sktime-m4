#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import operator
import os
from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from sktime.performance_metrics.forecasting import mase_loss
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.forecasting import check_y
from statsmodels.tsa.stattools import acf


def save_results(file, results):
    """Helper function to save results for during orchestration"""
    if not isinstance(results, list) and all(
            isinstance(r, dict) for r in results):
        raise ValueError("Expected input format is a list of dictionaries")

    df = pd.DataFrame(results)
    df = df.sort_values("id")

    # save predictions
    y_preds = pd.DataFrame(np.vstack(df.loc[:, "y_pred"]))
    y_preds.index = df.id
    y_preds.to_csv(file + "_y_pred.txt")

    # save metrics
    def _save_metrics(file, df, metric):
        values = df.loc[:, metric].values
        np.savetxt(file + f"_{metric}.txt", values)

    _save_metrics(file, df, "smape")
    _save_metrics(file, df, "mase")

    # save timings if available
    if all(col in df.columns for col in ["fit_time", "predict_time"]):
        timings = df.loc[:, ["fit_time", "predict_time"]].values
        np.savetxt(file + f"_timings.txt", timings)


def load_metadata(path):
    """Load metadata"""
    meta = pd.read_csv(os.path.join(path, "M4-info.csv"))
    assert meta.shape[0] == 100_000  # check that there are 100_000 series
    return meta


def load_mean_metrics(path, forecasters, datasets, metric):
    return pd.read_csv(os.path.join(path, f"{metric}_replicated.csv"),
                       index_col=0).reindex(forecasters).filter(
        datasets)


def load_published_mean_metrics(repo_path, datasets, forecasters):
    """Helper function to load and format published results"""
    m4_results = pd.read_excel(
        os.path.join(repo_path, "Evaluation and Ranks.xlsx"),
        sheet_name="Point Forecasts-Frequency",
        header=[0, 1]).dropna(axis=0)

    def _format_results(results, metric, datasets, forecasters):
        metrics = results.loc[:, ["Method", metric]]
        metrics.columns = metrics.columns.droplevel()
        metrics = metrics.set_index("User ID").drop(columns="Total")
        metrics = format_df(metrics, datasets, forecasters)
        return metrics

    smape = _format_results(m4_results, "sMAPE", datasets,
                            forecasters) / 100  # scaled by 100
    mase = _format_results(m4_results, "MASE", datasets, forecasters)
    owa = _format_results(m4_results, "OWA", datasets, forecasters)
    return smape, mase, owa


def M4_owa_loss(mase, smape, naive2_mase, naive2_smape):
    """overall weighted average of sMAPE and MASE loss used in M4 competition

    References
    ----------
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
    """
    return ((np.nanmean(smape) / np.mean(naive2_smape)) + (
            np.nanmean(mase) / np.mean(naive2_mase))) / 2


def get_metric_file_name(path, forecaster, dataset, metric):
    """Get file name"""
    return os.path.join(path, forecaster,
                        f"{forecaster}_{dataset}_{metric}.txt")


def save_replicated_metrics(values, path, forecaster, dataset, metric):
    file = get_metric_file_name(path, forecaster, dataset, metric)
    np.savetxt(file, values)


def load_replicated_metrics(path, forecaster, dataset, metric):
    """Load values for given forecaster, dataset and metric"""
    file = get_metric_file_name(path, forecaster, dataset, metric)
    return np.loadtxt(file)


def compute_and_save_mean_metrics(path, datasets, forecasters):
    """Helper function to load replicated metrics for individual series and
    compute average metrics, including OWA.
    """
    nans = np.zeros((len(forecasters), len(datasets), 3))
    results = []
    for d, dataset in enumerate(datasets):
        # load naive 2 losses to compute owa loss
        naive2_mase = load_replicated_metrics(path, "Naive2", dataset, "mase")
        naive2_smape = load_replicated_metrics(path, "Naive2", dataset,
                                               "smape")

        for f, forecaster in enumerate(forecasters):
            smape_replicated = load_replicated_metrics(path, forecaster,
                                                       dataset, "smape")
            mase_replicated = load_replicated_metrics(path, forecaster,
                                                      dataset, "mase")

            # replace infs with nans
            def replace_infs_with_nan(x):
                is_inf = np.logical_or(x == np.inf, x == -np.inf)
                x[is_inf] = np.nan
                return x

            smape_replicated = replace_infs_with_nan(smape_replicated)
            mase_replicated = replace_infs_with_nan(mase_replicated)

            # compute and save owa metrics
            owa_replicated = M4_owa_loss(mase_replicated, smape_replicated,
                                         naive2_mase, naive2_smape)
            save_replicated_metrics(np.array([owa_replicated]), path,
                                    forecaster, dataset, "owa")

            # count nan values
            nans[f, d, 0] = np.isnan(smape_replicated).sum()
            nans[f, d, 1] = np.isnan(mase_replicated).sum()

            # compute means, ignoring nan values if there are less than 1%
            # missing values
            if nans[f, d, 1] / len(mase_replicated) < 0.01:
                smape_mean = np.nanmean(smape_replicated)
                mase_mean = np.nanmean(mase_replicated)
            else:
                smape_mean = np.nan
                mase_mean = np.nan

            # compute mean metrics
            result = {
                "forecaster": forecaster,
                "dataset": dataset,
                "smape": smape_mean,
                "mase": mase_mean,
                "owa": owa_replicated
            }
            results.append(result)

    # write out nan counts
    smape_nans = pd.DataFrame(nans[:, :, 0], index=forecasters,
                              columns=datasets)
    format_df(smape_nans, datasets, forecasters).to_csv(
        os.path.join(path, f"smape_replicated_nan.csv"))

    mase_nans = pd.DataFrame(nans[:, :, 1], index=forecasters,
                             columns=datasets)
    format_df(mase_nans, datasets, forecasters).to_csv(
        os.path.join(path, f"mase_replicated_nan.csv"))

    # collect and format results
    results = pd.DataFrame(results)

    def _format_and_save_mean_metrics(path, results, metric):
        metrics = results.pivot(columns="dataset", index="forecaster",
                                values=metric)
        metrics = format_df(metrics, datasets, forecasters)
        metrics.to_csv(os.path.join(path, f"{metric}_replicated.csv"))
        return metrics

    smape_replicated = _format_and_save_mean_metrics(path, results, "smape")
    mase_replicated = _format_and_save_mean_metrics(path, results, "mase")
    owa_replicated = _format_and_save_mean_metrics(path, results, "owa")
    return smape_replicated, mase_replicated, owa_replicated


def compute_and_save_original_metrics(repo_path, results_path, forecasters):
    """Helper function to compute individual losses which are not in results
    file
    but which can be found in submission files"""
    datadir = os.path.join(repo_path, "Dataset")
    meta = load_metadata(datadir)
    sps = meta.set_index("SP")["Frequency"].to_dict()

    for forecaster in forecasters:
        print(f"Forecaster: {forecaster} ...")

        filedir = os.path.join(results_path, forecaster)
        if not os.path.isdir(filedir):
            print(f"Creating results folder: {filedir} ...")
            os.makedirs(filedir)

        # load original predictions
        predictions = pd.read_csv(os.path.join(repo_path, "Point Forecasts",
                                               f"submission-{forecaster}.csv"))

        # compute scores from original predictions
        datasets = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily",
                    "Hourly"]
        for dataset in datasets:
            print(f"Dataset: {dataset} ...")

            # if results file already exists, skip series
            file = os.path.join(filedir, f"{forecaster}_{dataset}")
            if os.path.isfile(f"{file}_y_pred.txt"):
                print(f"{forecaster} skipped: forecasts already exists")
                continue

            test = pd.read_csv(
                os.path.join(datadir, "Test", f"{dataset}-test.csv"))
            train = pd.read_csv(
                os.path.join(datadir, "Train", f"{dataset}-train.csv"))

            Y_pred = predictions.loc[
                     predictions["id"].str.startswith(dataset[0]), :].dropna(
                axis=1).set_index(
                "id").sort_index()
            Y_test = test.set_index("V1").sort_index()
            Y_train = train.set_index("V1").sort_index()

            assert Y_pred.shape == Y_test.shape
            assert Y_test.shape[0] == Y_train.shape[0]
            assert np.array_equal(Y_test.index.values, Y_pred.index.values)
            assert np.array_equal(Y_test.index.values, Y_train.index.values)

            index = Y_train.index

            train_index = np.arange(Y_train.shape[1])
            test_index = np.arange(train_index[-1],
                                   train_index[-1] + Y_pred.shape[1]) + 1

            Y_train.columns = train_index
            Y_test.columns = test_index
            Y_pred.columns = test_index

            sp = sps[dataset]

            results = []
            for i, idx in enumerate(index):
                y_pred = Y_pred.iloc[i, :]
                y_test = Y_test.iloc[i, :]
                y_train = Y_train.iloc[i, :].dropna()

                # compute metrics, owa is computed afterwards
                mase = mase_loss(y_test, y_pred, y_train=y_train, sp=sp)
                smape = smape_loss(y_test, y_pred)

                result = {
                    "id": idx,
                    "y_pred": y_pred,
                    "mase": mase,
                    "smape": smape
                }
                results.append(result)

            save_results(file, results)


def seasonality_test_R(y, sp):
    """Seasonality test used in M4 competition

    R and Python versions were inconsistent [2], this is the Python
    translation of the R version [1].

    References
    ----------
    ..[1]  https://github.com/Mcompetitions/M4-methods/blob/master
    /Benchmarks%20and%20Evaluation.R
    ..[2]  https://github.com/Mcompetitions/M4-methods/issues/25
    """
    y = check_y(y)
    y = np.asarray(y)
    n_timepoints = len(y)

    sp = check_sp(sp)
    if sp == 1:
        return False

    if n_timepoints < 3 * sp:
        warn(
            "Did not perform seasonality test, as `y`` is too short for the "
            "given `sp`, returned: False")
        return False

    else:
        coefs = acf(y, nlags=sp, fft=False)  # acf coefficients
        coef = coefs[sp]  # coefficient to check

        tcrit = 1.645  # 90% confidence level
        limits = tcrit / np.sqrt(n_timepoints) * np.sqrt(
            np.cumsum(np.append(1, 2 * coefs[1:] ** 2)))
        limit = limits[sp - 1]  #  zero-based indexing
        return np.abs(coef) > limit


def seasonality_test_Python(y, sp):
    """Seasonality test used in M4 competition

    R and Python versions were inconsistent [2], this is a copy of the
    Python version [1].

    References
    ----------
    ..[1]  https://github.com/M4Competition/M4-methods/blob/master
    /ML_benchmarks.py
    ..[2]  https://github.com/Mcompetitions/M4-methods/issues/25
    """

    if sp == 1:
        return False

    def _acf(data, k):
        m = np.mean(data)
        s1 = 0
        for i in range(k, len(data)):
            s1 = s1 + ((data[i] - m) * (data[i - k] - m))

        s2 = 0
        for i in range(0, len(data)):
            s2 = s2 + ((data[i] - m) ** 2)

        return float(s1 / s2)

    s = _acf(y, 1)
    for i in range(2, sp):
        s = s + (_acf(y, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(y)))

    return (abs(_acf(y, sp))) > limit


def compute_and_save_difference_in_metric(replicated_results_path,
                                          original_results_path, datasets,
                                          forecasters,
                                          metric):
    """Helper function to compute differences in metrics"""
    original = load_mean_metrics(original_results_path, forecasters, datasets,
                                 metric)
    replicated = load_mean_metrics(replicated_results_path, forecasters,
                                   datasets, metric)

    def _compute_difference(replicated, original):
        return replicated - original

    diff = _compute_difference(replicated, original)
    diff = format_df(diff, datasets, forecasters)
    diff.to_csv(os.path.join(replicated_results_path, f"{metric}_diff.csv"))

    # percentage difference
    diff = _compute_difference(replicated, original)
    perc_diff = diff / original * 100
    perc_diff = format_df(perc_diff, datasets, forecasters)
    perc_diff.to_csv(
        os.path.join(replicated_results_path, f"{metric}_perc_diff.csv"))


def compute_and_save_difference_in_metrics(replicated_results_path,
                                           original_results_path, datasets,
                                           forecasters):
    """Helper function to compute differences in metrics"""
    from config import METRICS
    for metric in METRICS:
        compute_and_save_difference_in_metric(replicated_results_path,
                                              original_results_path, datasets,
                                              forecasters,
                                              metric)


def find_forecasters(path, datasets):
    """Helper function to get forecaster names from existing results files"""

    def is_dir(path, file):
        return os.path.isdir(os.path.join(path, file))

    def is_complete(path, file):
        """Check if files for all datasets exist"""
        return all([any(
            [dataset in file for file in os.listdir(os.path.join(path, file))])
            for dataset in datasets])

    forecasters = [file for file in os.listdir(path) if
                   is_dir(path, file) and is_complete(path, file)]
    if len(forecasters) == 0:
        raise ValueError("No completed forecasters found.")
    return forecasters


def format_df(df, datasets, forecasters):
    """Re-order index and columns"""
    df.columns.name = None
    df.index.name = None
    df.index = df.index.astype(str)
    return df.filter(datasets).reindex(forecasters).dropna()


def load_all_metrics(path, forecasters, datasets, metric):
    res = dict()
    for forecaster in forecasters:
        scores = []
        for dataset in datasets:
            file = os.path.join(path, forecaster,
                                f"{forecaster}_{dataset}_{metric}.txt")
            score = np.loadtxt(file)
            scores.append(score)
        res[forecaster] = np.hstack(scores)

    return pd.DataFrame(res)


def load_timings(path, forecasters, datasets):
    timings = np.zeros((len(forecasters), len(datasets)))
    for f, forecaster in enumerate(forecasters):
        for d, dataset in enumerate(datasets):
            file = os.path.join(path, forecaster,
                                f"{forecaster}_{dataset}_timings.txt")
            timings[f, d] = np.loadtxt(file).sum() / 60  # scale to minutes

    df = pd.DataFrame(timings, index=forecasters, columns=datasets)
    df["Total"] = df.sum(axis=1)
    return df


def compute_timings(path, forecasters, datasets):
    timings = load_timings(path, forecasters, datasets)

    # scale replicated results to number of CPUs used to measure published
    # run time values
    n_cpus_published = 8
    n_cpus_replicated = 32
    timings["replicated"] = timings[
                                "Total"] / n_cpus_replicated * n_cpus_published
    return timings


def compute_overall_metrics(path, forecasters, datasets):
    """Compute overall average as average weighted by number of series"""
    # get weights
    from config import DATADIR, METRICS
    meta = load_metadata(DATADIR)
    weights = meta.loc[:, "SP"].value_counts().reindex(datasets)

    # load mean metrics
    metrics = []
    for metric in METRICS:
        m = load_mean_metrics(path, forecasters, datasets, metric)
        metrics.append(m)

    # get index and columns
    index = metrics[0].index  # forecasters
    columns = metrics[0].columns  # datasets

    # preallocate array
    avg_metrics = np.zeros((len(forecasters), len(metrics)))

    # check weights
    np.testing.assert_array_equal(weights.index, columns)

    # compute weighted average
    for i, metric in enumerate(metrics):
        assert metric.index.equals(index)
        avg_metrics[:, i] = np.average(metric, axis=1, weights=weights)

    avg_metrics = pd.DataFrame(avg_metrics, index=forecasters, columns=METRICS)
    avg_metrics["smape"] = avg_metrics["smape"] * 100
    return avg_metrics


def wilcoxon_holm(df_perf, alpha=0.05):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and
    then use Holm
    to reject the null's hypothesis

    References
    ----------
    Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
            Germain Forestier <germain.forestier@uha.fr>
            Jonathan Weber <jonathan.weber@uha.fr>
            Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
            Pierre-Alain Muller <pierre-alain.muller@uha.fr>
    License: GPL3
    https://github.com/hfawaz/cd-diagram
    """
    # print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be
        # rejected
        print(
            'the null hypothesis over the entire classifiers cannot be '
            'rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon
    # signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(
            df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
            , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(
                df_perf.loc[df_perf['classifier_name'] == classifier_2]
                ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (
                p_values[i][0], p_values[i][1], p_values[i][2], True)
    #         else:
    #             # stop
    #             break
    # compute the average ranks to be returned (useful for drawing the cd
    # diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(
        classifiers)].sort_values(
        ['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m,
                                                             max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers),
                            columns=
                            np.unique(sorted_df_perf['dataset_name']))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(
        ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets
