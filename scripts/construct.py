#!/usr/bin/env python3 -u

__author__ = "Markus Löning"
__all__ = ["construct_forecasters"]

import os
from contextlib import redirect_stderr

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose import RecursiveRegressionForecaster
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.adapt import \
    SingleSeriesTransformAdaptor
from sktime.transformers.single_series.boxcox import BoxCoxTransformer
from sktime.transformers.single_series.detrend import ConditionalDeseasonalizer
from sktime.transformers.single_series.detrend import Detrender
from xgboost import XGBRegressor

from utils import seasonality_test_Python
from utils import seasonality_test_R

with redirect_stderr(open(os.devnull, "w")):
    from sktime_dl.deeplearning import SimpleRNNRegressor

PARAM_GRID = {"window_length": np.array([3, 4, 6, 8, 10, 12, 15, 18, 21, 24],
                                        dtype=np.int)}
N_ESTIMATORS = 500
SEASONAL_MODEL = "multiplicative"
REGRESSORS = {
    "LR": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=1),
    "RF": RandomForestRegressor(n_estimators=N_ESTIMATORS),
    "XGB": XGBRegressor(n_estimators=N_ESTIMATORS),
}

ses = ExponentialSmoothing()
holt = ExponentialSmoothing(trend="add", damped=False)
damped = ExponentialSmoothing(trend="add", damped=True)


def make_pipeline(*estimators):
    """Helper function to make pipeline"""
    steps = [(estimator.__class__.__name__, estimator) for estimator in
             estimators]
    return TransformedTargetForecaster(steps)


def deseasonalise(forecaster, seasonality_test=seasonality_test_R, **kwargs):
    return make_pipeline(
        ConditionalDeseasonalizer(seasonality_test, **kwargs),
        forecaster
    )


def boxcox(forecaster):
    return make_pipeline(
        BoxCoxTransformer(bounds=(0, 1)),
        forecaster
    )


def deseasonalise_boxcox(forecaster, seasonality_test=seasonality_test_R,
                         **kwargs):
    return make_pipeline(
        ConditionalDeseasonalizer(seasonality_test, **kwargs),
        BoxCoxTransformer(bounds=(0, 1)),
        forecaster
    )


def construct_ml_forecasters(sp, fh):
    WINDOW_LENGTH = np.maximum(3, sp)

    def make_ml_pipeline(regressor):
        return make_pipeline(
            Detrender(
                PolynomialTrendForecaster(degree=1, with_intercept=True)),
            # linear detrend
            SingleSeriesTransformAdaptor(StandardScaler()),  # scale
            RecursiveRegressionForecaster(regressor=regressor,
                                          window_length=WINDOW_LENGTH))

    forecasters = {name: make_ml_pipeline(regressor) for name, regressor in
                   REGRESSORS.items()}
    return forecasters


def construct_ml_forecasters_s(sp, fh):
    ml_forecasters = construct_ml_forecasters(sp, fh)
    forecasters = {
        f"{name}-s": deseasonalise(forecaster,
                                   seasonality_test=seasonality_test_R, sp=sp,
                                   model=SEASONAL_MODEL)
        for name, forecaster in ml_forecasters.items()
    }
    return forecasters


def make_cv(fh):
    return SingleWindowSplitter(fh=fh)


def construct_tuned_ml_forecasters(sp, fh):
    cv = make_cv(fh)

    def make_tuned_ml_pipeline(regressor):
        return make_pipeline(
            Detrender(
                PolynomialTrendForecaster(degree=1, with_intercept=True)),
            # linear detrend
            SingleSeriesTransformAdaptor(StandardScaler()),  # scale
            ForecastingGridSearchCV(
                RecursiveRegressionForecaster(regressor=regressor),
                cv=cv, param_grid=PARAM_GRID)
        )

    forecasters = {
        f"{name}-t": make_tuned_ml_pipeline(regressor)
        for name, regressor in REGRESSORS.items()
    }
    return forecasters


def construct_ml_forecasters_t_s(sp, fh):
    ml_forecasters = construct_tuned_ml_forecasters(sp, fh)
    forecasters = {
        f"{name}-s": deseasonalise(forecaster)
        for name, forecaster in ml_forecasters.items()
    }
    return forecasters


def construct_M4_forecasters(sp, fh):
    kwargs = {"model": SEASONAL_MODEL, "sp": sp} if sp > 1 else {}

    theta_bc = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_R,
                                  **kwargs),
        BoxCoxTransformer(bounds=(0, 1)),
        ThetaForecaster(deseasonalise=False)
    )
    MLP = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_Python,
                                  **kwargs),
        Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
        RecursiveRegressionForecaster(
            regressor=MLPRegressor(hidden_layer_sizes=6, activation="identity",
                                   solver="adam", max_iter=100,
                                   learning_rate="adaptive",
                                   learning_rate_init=0.001),
            window_length=3)
    )
    RNN = make_pipeline(
        ConditionalDeseasonalizer(seasonality_test=seasonality_test_Python,
                                  **kwargs),
        Detrender(PolynomialTrendForecaster(degree=1, with_intercept=True)),
        RecursiveTimeSeriesRegressionForecaster(
            regressor=SimpleRNNRegressor(nb_epochs=100),
            window_length=3)
    )

    forecasters = {
        "Naive": NaiveForecaster(strategy="last"),
        "sNaive": NaiveForecaster(strategy="seasonal_last", sp=sp),
        "Naive2": deseasonalise(NaiveForecaster(strategy="last"), **kwargs),
        "SES": deseasonalise(ses, **kwargs),
        "Holt": deseasonalise(holt, **kwargs),
        "Damped": deseasonalise(damped, **kwargs),
        "Theta": deseasonalise(ThetaForecaster(deseasonalise=False), **kwargs),
        "ARIMA": AutoARIMA(suppress_warnings=True, error_action="ignore",
                           sp=sp),
        "Com": deseasonalise(EnsembleForecaster(
            [("ses", ses), ("holt", holt), ("damped", damped)]), **kwargs),
        "MLP": MLP,
        "RNN": RNN,
        "260": theta_bc,
    }
    return forecasters


def construct_theta_boosted_forecasters(sp, fh):
    def theta_boosted(regressor):
        WINDOW_LENGTH = np.maximum(3, sp)
        return make_pipeline(
            ConditionalDeseasonalizer(seasonality_test=seasonality_test_R,
                                      sp=sp, model=SEASONAL_MODEL),
            BoxCoxTransformer(bounds=(0, 1)),
            Detrender(ThetaForecaster(deseasonalise=False)),
            SingleSeriesTransformAdaptor(StandardScaler()),
            RecursiveRegressionForecaster(regressor=regressor,
                                          window_length=WINDOW_LENGTH)
        )

    regressors = {name: regressor for name, regressor in REGRESSORS.items() if
                  name in ["KNN", "RF", "XGB"]}
    forecasters = {f"{name}_theta": theta_boosted(regressor) for
                   name, regressor in regressors.items()}
    return forecasters


def construct_theta_boosted_forecasters_t(sp, fh):
    def theta_boosted_tuned(regressor):
        cv = make_cv(fh)
        return make_pipeline(
            ConditionalDeseasonalizer(seasonality_test=seasonality_test_R,
                                      sp=sp, model=SEASONAL_MODEL),
            BoxCoxTransformer(bounds=(0, 1)),
            Detrender(ThetaForecaster(deseasonalise=False)),
            SingleSeriesTransformAdaptor(StandardScaler()),
            ForecastingGridSearchCV(RecursiveRegressionForecaster(
                regressor=regressor), param_grid=PARAM_GRID, cv=cv)
        )

    regressors = {name: regressor for name, regressor in REGRESSORS.items() if
                  name in ["KNN", "RF", "XGB"]}
    forecasters = {f"{name}_theta_t": theta_boosted_tuned(regressor) for
                   name, regressor in regressors.items()}
    return forecasters


def construct_forecasters(sp, fh):
    construct_funcs = [
        construct_M4_forecasters,
        construct_ml_forecasters,
        construct_ml_forecasters_s,
        construct_ml_forecasters_t_s,
        construct_theta_boosted_forecasters,
        construct_theta_boosted_forecasters_t
    ]

    # construct forecasters
    forecasters_dicts = []
    for func in construct_funcs:
        forecasters = func(sp, fh)
        forecasters_dicts.append(forecasters)

    # combine into single dictionary
    all_forecasters = dict()
    for forecaster_dict in forecasters_dicts:
        for name, forecaster in forecaster_dict.items():
            if name in all_forecasters.keys():
                raise KeyError(f"{name} already defined")
            all_forecasters[name] = forecaster

    return all_forecasters
