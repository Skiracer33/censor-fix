import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
import pkg_resources
import censorfix

stan_dir = os.path.dirname(__file__) +'/../stan_compiled/'

class censorImputer():

    def __init__(self,
                 sample_posterior=True,
                 column_choice='auto',
                 initial_point='auto',
                 distribution='gaussian',
                 missing_values=np.NAN,
                 no_columns='all',
                 max_iter=5,
                 stan_iterations=4000,
                 debug=True,
                 n_jobs=8,
                 imputation_order='ascending',
                 number_imputations=1):
        """
        Multivariate imputer that handles censored values

        This is a strategy for dealing with missing and censored values in data sets. It can handle both lower and upper censoring
        points.

        Parameters
        ----------
        sample_posterior : bool
            whether to use the best prediction at each step or a bayesian imputation
        distribution : gaussian, t-distribution, skew normal
            the distribution to use for the experiment 
        missing_values : str
            the placeholder for missing vlaues these will be imputed
        max_iter : int
            the number of cycles
        no_columns : int
            how many columns to use for the imputation
        stan_iterations : int
            number of iterations for Stan to run default works well
        imputation_order : str ascending
            the order of imputations 
        debug: bool
            display debug information
        number_imputations : int
            the number of imputations required
        """

        if not os.path.isfile(os.path.dirname(__file__) +'/../stan_compiled/s3.stan'):
            print('Compiling stan code')
            censorfix.compile_stan.compile_all()
        self.sample_posterior = sample_posterior
        self.column_choice = column_choice
        self.initial_point = initial_point
        self.distribution = distribution
        self.no_columns = no_columns
        self.max_iter = max_iter
        self.stan_iterations = stan_iterations
        self.data = None
        self.debug = debug
        self.n_jobs = n_jobs
        self.number_imputations=number_imputations
        if distribution == 'gaussian':
            self.stan_model = joblib.load(stan_dir + 's3.stan')
        if distribution == 'skew normal distribution':
            self.stan_model = joblib.load(stan_dir + 's4.stan') 
        if distribution == 't_distribution':
            self.stan_model = joblib.load(stan_dir + 's5.stan') 

        if not sample_posterior and number_imputations!=1:
            print('error posterior sampling needs to be enabled if doing multiple imputation ')
            return 
        
    def impute_once(self, y, X, U, L):
        """
        impute one column of censored values using STAN program with chosen options

        Parameters
        ----------
        y : array like
            censored values
        X : array like
            independent values
        U : double 
            the upper censored values
        L : double
            the lower censored values
        """

        for i in range(len(y) - 1):  # check if y is sorted
            if y.iloc[i] > y.iloc[i + 1]:
                print('values need to be sorted')
                return y

        K = X.shape[1]
        N_cens_right = sum(y >= U) if U != 'NA' else 0
        if L != 'NA':
            N_cens_left = sum(y <= L)
        else:
            N_cens_left = 0
            L = -np.inf
        if N_cens_right == 0 and N_cens_left == 0:
            return y  # nothing to impute
        N_obs = X.shape[0] - N_cens_right - N_cens_left

        #Feed the data into stan using a dictionary
        data = {'N_obs': N_obs,
                'N_cens_left': N_cens_left,
                'N_cens_right': N_cens_right,
                'x_obs': X.values,
                'y_obs': y[N_cens_left:N_obs + N_cens_left].values,
                'U': U,
                'L': L,
                'K': K}
        res = self.stan_model.sampling(
            data=data,
            iter=self.stan_iterations,
            n_jobs=self.n_jobs)
        if self.debug:
            print(res.stansummary())

        if self.sample_posterior:
            try:
                y[N_cens_left + N_obs:] = res.extract()['y_cens_right'][-1]
            except KeyError:
                pass
            try:
                y[:N_cens_left] = res.extract()['y_cens_left'][-1]
            except KeyError:
                pass
        else:
            try:
                y[N_cens_left + N_obs:] = res.extract()['y_cens_right'].mean(axis=0)
            except KeyError:
                pass
            try:
                y[:N_cens_left] = res.extract()['y_cens_left'].mean(axis=0)
            except KeyError:
                pass
        return y

    def impute(self, data, right_cen=None, left_cen=None, iter_val=1):
        """
        impute multiple columns in an iterative style

        
        returns the data in a sorted form 
        if multiple imputations are requested data is returned 

        Parameters
        ----------
        data : pandas dataframe
            the data as a pandas dataframe
        right_cen : a list of doubles
            the right censoring points of the data NA if no censoring
        left_cen : list of doubles 
            of the left censoring points of the data NA if no censoring
        iter_val : int
            the number of imputation rounds to perform
        Returns
        -------
        array
            Dataset of with imputed values
        """
        no_features = data.shape[1]
        
        if not right_cen or not left_cen:
            print('no censoring values provided')
            return 

        if not right_cen:
            right_cen=['NA']*no_features

        if not left_cen:
            left_cen=['NA']*no_features

        if not isinstance(data, pd.DataFrame):
            print("data needs to be in a pandas dataframe")
        
        def select_columns(i):  # selects which columns to use
            if self.no_columns == 'all':
                return list(range(i)) + list(range(i + 1, no_features))
            return 'error not implemented yet' #TODO

        #single imputations
        if self.number_imputations==1:
            for _ in tqdm(range(iter_val)):
                for i in range(no_features):
                    data = data.sort_values(by=data.columns[i], ascending=True)
                    data.iloc[:, i] = self.impute_once(data.iloc[:, i],
                                                    data.iloc[:, select_columns(i)],
                                                    right_cen[i], left_cen[i])
            return data

        #multiple imputations
        else:
            ret=[]
            for _ in tqdm(range(iter_val-1)):
                for i in range(no_features):
                    data = data.sort_values(by=data.columns[i], ascending=True)
                    data.iloc[:, i] = self.impute_once(data.iloc[:, i],
                                                data.iloc[:, select_columns(i)],
                                                right_cen[i], left_cen[i])
            i=0
            for j in range(self.number_imputations):         
                ret.append(data.sort_values(by=data.columns[i], ascending=True).copy()) #create the imputations
            for data in ret:
                data = self.impute_once(data.iloc[:, i],
                                        data.iloc[:, select_columns(i)],
                                        right_cen[i], left_cen[i])
            for i in range(1,no_features):
                for data in ret:
                     data = data.sort_values(by=data.columns[i], ascending=True)
                     data.iloc[:, i] = self.impute_once(data.iloc[:, i],
                                                data.iloc[:, select_columns(i)],
                                                right_cen[i], left_cen[i])
            return ret

