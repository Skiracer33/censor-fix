import numpy as np
import pandas as pd
import joblib
from censorfix import censorfix


def create_data():
    """
    returns two dataframes a copy of one another
    """
    c = 0.5
    n = 3
    cov = c + np.identity(n) * (1 - c)
    size = 100
    full_data = np.random.multivariate_normal(
        [0 for i in range(n)], cov, size=size)
    df = pd.DataFrame(full_data)
    df2 = df.copy()
    return df, df2


def single_dim_test():
    """
    Test censorfix in one d 
    with a gaussian distribution of data
    """
    df, df2 = create_data() 
    censor_high = 1.5
    censor_low =- 0.5
    df.loc[df[0] > censor_high, 0] = censor_high
    df.loc[df[0] < censor_low, 0] = censor_low
    imp = censorfix.censorImputer(
        debug=False, no_columns=2, sample_posterior=True)
    df = df.sort_values(by=0, ascending=True)
    imp.impute_once(df[0], df[[1, 2]], censor_high, censor_low)
    fig, ax = plt.subplots(1, 1)
    df2.plot(kind='scatter', x=0, y=2, ax=ax, color='pink',label='imputed')
    df.plot(kind='scatter', x=0, y=2, ax=ax,label='true')
    plt.title('single imputation of censored values')
    plt.show()
    return df,df2


def multi_imp_test(plot=True):
    """
    Tests the creation of multiple imputations
    plots results or returns dataframe and the imputed dat
    assumes gaussian distribution
    """
    df, df2 = create_data()
    # censor the first dataframe
    censor_high_1=0.8
    censor_high_2=1
    censor_low_1=-0.6
    censor_low_2=-2
    df.loc[df[0] > censor_high_1, 0] = censor_high_1
    df.loc[df[0] < censor_low_1, 0] = censor_low_1
    df.loc[df[1] > censor_high_2, 1] = censor_high_2
    df.loc[df[1] < censor_low_2, 1] = censor_low_2

    imp = censorfix.censorImputer(
        debug=False, sample_posterior=True,number_imputations=3)
    U = [censor_high_1, censor_high_2, 'NA']  # the upper censor values
    L = [censor_low_1, censor_low_2, 'NA'] # the lower censor values

    data_mi = imp.impute(df, U, L, iter_val=2)

    if plot:
        fig, ax = plt.subplots(1, 1)
        colours=['red','yellow','green']
        for i,data in enumerate(data_mi):
            data.plot(kind='scatter',x=0,y=1,color=colours[i],label='imputation {}'.format(i),ax=ax)
        df2.plot(kind='scatter',x=0,y=1,color='blue',label='original',ax=ax)
        plt.title('Multiple imputations comparison')
        plt.legend()
        plt.show()
    return df2, data_mi


def multi_dim_test():
    """
    Test censorfix for multiple imputation of multiple dimensions
    gaussian distribution
    """
    df, df2 = create_data() 

    # censor the first dataframe
    censor_high_1=0.8
    censor_high_2=0.5
    censor_low_1=-0.3
    censor_low_2=-0.7
    df.loc[df[0] > censor_high_1, 0] = censor_high_1
    df.loc[df[0] < censor_low_1, 0] = censor_low_1
    df.loc[df[1] > censor_high_2, 1] = censor_high_2
    df.loc[df[1] < censor_low_2, 1] = censor_low_2

    imp = censorfix.censorImputer(
        debug=False, sample_posterior=True)
    U = [censor_high_1, censor_high_2, 'NA']  # the upper censor values
    L = [censor_low_1, censor_low_2, 'NA'] # the lower censor values
 
    fig, ax = plt.subplots(1, 1)
    df.plot(kind='scatter', x=0, y=1, ax=ax, color='yellow', label='censored')
    df = imp.impute(df, U, L, iter_val=2)
    df2.plot(
        kind='scatter',
        x=0,
        y=1,
        ax=ax,
        color='pink',
        label='imputed_values')
    df.plot(kind='scatter', x=0, y=1, ax=ax, label='actual')
    plt.legend()
    plt.title('Multivariate Censor Imputation')
    plt.show()
    return df,df2

