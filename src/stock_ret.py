# authur: Zicheng Xiao
# date: 2022-12-18
# version: 1.0

"""This module is to create a class named stock_ret. It provides
- a public class ```stock``` with the following attributes:
    - ```stock_name```: a string
    - ```stock_price```: a float
    - ```stock_quantity```: an integer
    - ```stock_date```: a string
    it provides the result of the stock returns, like daily return, monthly return, and annual return.

- a public class ```portfolio``` with the following attributes:
it designed to easily calculate the portfolio returns, easily manage certain financial portfoliom and make the most common quatitative calculations, such as 
    - cumulative returns of the portfolio's stocks, like daily return, monthly return, and annual return.
    - the portfolio's volatility
    - the portfolio's sharpe ratio
    - the portfolio's beta
    - the skewness and kurtosis of the portfolio's returns
    - the portfolio's alpha

- a public class ```Fama French 3/5 factor``` with the following attributes
:
    

- a public class ```DTGW``` with the following attributes:
    """

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class stock_ret(object):
    """Object that contains information about a stock/fund.
    To initialize the object, it requires a name, information about the stock/fund. To initialize the object, it requires a name, information about the stock/fund given as one of the following data structures:
    - pandas series
    - pandas dataframe
    The investment information can contain as little information as its name, and the amount invested in it the column labels must be name and allocation respectively, but it can also contain more information such as
    - Year
    - Strategy
    - CCY
    - etc
    It also requires either data, e.g. daily closing prices as a ''pandas.DataFrame'' or ''pandas.Series''
    ''data'' must be given as a ''pandas.DataFrame'', and at least one data column contain one column label ''<stock_name> - Adj. Close'' which is used to compute the return of investment. However, ``data`` can contain more data in addition columns''"""

    def __init__(self, investmentinfo, data=None):
        """Investmentinfo: ``pandas.DataFrame`` of investment information
        data: ``pandas.DataFrame`` of stock price
        """
        self.name = investmentinfo.name
        self.investmentinfo = investmentinfo
        self.data = data
        # compute expected return and volatility of stock
        self.volatility = self.compute_volatility()
        self.skew = self.compute_skew()
        self.kurtosis = self.compute_kurtosis()

    # functions to compute quatities
    def comp_daily_returns(self, data):
        """Compute the daily returns of the stock
            :math:`latex: $$ Ret = \frac{price_{t_i} - price_{t_{i-1}} + div_{t}}{price_{t_{i-1}}} $$`
        :Output:
            : daily returns: Daily returns of the stock
        """
        return data.pct_change().dropna(how='all').replace([np.inf, -np.inf], np.nan)

    # functions to compute quatities
    def compute_volatility(self, freq=252):
        """Compute the volatility of the stock
        :Input:
            : freq: ``int`` (default: ``252``), number of trading days in a year, default
                  value corresponds to trading days in a year
        :Output:
            : expected return: Expected return of the stock
        """
        return self.comp_daily_returns().std()*np.sqrt(freq)

    def cumulative_returns(self, dividend=0):
        """Returns DataFrame with cumulative returns
        :math: CR = \frac{P_{t_i} - P_{t_0} + Div_{t}}{P_{t_0}}
        Input:
            data: ``pandas.DataFrame`` of stock price
            dividend: ``pandas.DataFrame`` of dividend
        Output:
            :ret: a ``pandas.DataFrame`` of cumulative returns of given stock returns.
        """
        return self.data.dropna(axis=0, how='any').apply(lambda x: (x - x[0] + dividend) / x[0])

    def daily_log_returns(self, dividend=0):
        """Returns DataFrame with daily log returns
        :math: DR = ln(1 + \frac{Price_{t_i}) - Price_{t_{i-1}}{Price_{t_{i-1}}})
        Input:
            data: ``pandas.DataFrame`` of stock price
            dividend: ``pandas.DataFrame`` of dividend
        Output:"""
        return self.data.dropna(axis=0, how='any').apply(lambda x: np.log(x) - np.log(x.shift(1)) + np.log(dividend))

    def historical_mean_return(self, data, freq=252):
        """Returns the mean return based on historical stock price data.
        : Input
            : data: ``pandas.DataFrame`` with daily stock price
            : freq: ``int`` (default: ``252``), number of trading days in a year, default
                  value corresponds to trading days in a year
            : Output
                : ret: a ``pandas.DataFrame`` of historical mean Returns.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be a pandas.DataFrame')
        return self.comp_daily_returns(data).mean() * freq

    def _comp_skew(self):
        """Compute and returns the skewness of the stock.
        """
        return self.data.skew().values[0]
    
    def _comp_kurtosis(self):
        """Compute and returns the kurtosis of the stock.
        """
        return self.data.kurtosis().values[0]
    
    def properties(self):
        """Nicely prints out the properties of the stock: Expected return, volatility, skewness, kurtosis as well as the ``Allocation``
        (and other information provided in investmentinfo.)"""
        # nicely printing out infomration and quantities of the stock
        string = "-"*50
        string += "\nStock: {}".format(self.name)
        string += "\nExpected return: {:.3f}%".format(self.expected_return.values[0])
        string += "\nVolatility: {:.3f}%".format(self.volatility.values[0])
        string += "\nSkewness: {:.5f}".format(self.skew)
        string += "\nKurtosis: {:.5f}".format(self.kurtosis)
        string += "\nInformation:"
        string += "\n" + str(self.investmentinfo.to_frame().transpose())
        string += "\n"
        string += "-" * 50
        print(string)

    