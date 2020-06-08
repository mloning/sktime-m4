# Replicating and Extending the M4 Study with sktime


This is a companion repository for our paper: 

* [Markus Löning, Franz Király: “Forecasting with sktime: Designing sktime's New Forecasting API and Applying It to Replicate and Extend the M4 Study”, 2020; arXiv:2005.08067](https://arxiv.org/abs/2005.08067)

The repository contains the code for reproducing our results. Our code is heavily based on [sktime](https://github.com/alan-turing-institute/sktime).

* `Makefile` contains convenience commands for running the replication and
 extension from the command line,
* `scripts/` contains our code for replicating and extending the M4 study,
* `results/replicated/` contains our obtained predictive performance results,
* `requirements.txt` contains the list of required packages to run our scripts.

To reproduce our results, we recommend to download the [original repository of
 the M4 study](https://github.com/Mcompetitions/M4-methods), particularly we
  use 

* the [M4 data set](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset),  
* the individual [point forecast results](https://github.com/Mcompetitions/M4-methods/tree/master/Point%20Forecasts) for comparison,
* the [summary results file](https://github.com/Mcompetitions/M4-methods/blob/master/Evaluation%20and%20Ranks.xlsx) to cross-check results.

In addition, you may have to adjust paths for the input and output
 directories in `scripts/config.py`. 
