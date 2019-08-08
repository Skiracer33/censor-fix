## CensorFix

[![Documentation Status](https://readthedocs.org/projects/censor-fix/badge/?version=latest)](https://censor-fix.readthedocs.io/en/latest/?badge=latest)


Please look at the website https://censor-fix.readthedocs.io/en/latest/


A library for multiple imputation of censored data.

This software performs multiple imputation on censored data using the probabilistic programming langauge (ppl) stan. 
For 1d imputation the library can fits a distribution to data to create imputations. 

For missing data with many features the library can model the censored values using the other features in a round robin fashion

How to use:

The data needs to be in a pandas dataframe and the columns that need imputing must be specified.
The data is assumed to have a normal distribution unless stated otherwise but there are options for the data to be uniform,t distributed or exponential. To make the data more normal you can try using sklearn.reprocessing.power_transform to make the data normal. 



