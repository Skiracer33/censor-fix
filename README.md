## CensorFix

[![Documentation Status](https://readthedocs.org/projects/censor-fix/badge/?version=latest)](https://censor-fix.readthedocs.io/en/latest/?badge=latest)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Example Notebooks and API and documentation are available [here](https://censor-fix.readthedocs.io/en/latest/)


### A library for multiple imputation of censored data.

This software performs multiple imputation on censored data using the probabilistic programming langauge Stan. 
The library fits a distributions to censored data and produces imputations based on the results.

How to use:

The data is assumed to have a normal distribution unless stated otherwise but there are options for the data to be uniform,t-distributed or exponential.Consider preprocessing using a power transform such as sklearn.reprocessing.power_transform before applying the algorithm.



