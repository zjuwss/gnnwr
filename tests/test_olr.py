import unittest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class OLS_SM:
    """
    `OLS` is the class to calculate the OLR coefficients of data.Get the coefficient by `object.params`.

    This class is used by statsmodels lib.

    :param dataset: Input data
    :param xName: the independent variables' column
    :param yName: the dependent variable's column
    """

    def __init__(self, dataset, xName: list, yName: list):
        self.__dataset = dataset
        self.__xName = xName
        self.__yName = yName
        self.__formula = yName[0] + '~' + '+'.join(xName)
        self.__fit = sm.formula.ols(self.__formula, dataset).fit()
        self.params = list(self.__fit.params.to_dict().values())
        intercept = self.__fit.params.iloc[0]
        self.params = self.params[1:]
        self.params.append(intercept)


class OLS_SKLEARN:
    """
    `OLS` is the class to calculate the OLR coefficients of data.Get the coefficient by `object.params`.

    This class is used by scikit-learn lib.

    :param dataset: Input data
    :param xName: the independent variables' column
    :param yName: the dependent variable's column
    """

    def __init__(self, dataset, xName: list, yName: list):
        self.__dataset = dataset
        self.__xName = xName
        self.__yName = yName
        self.__fit = LinearRegression(fit_intercept=True)
        y = self.__dataset[self.__yName[0]] if len(
            self.__yName) == 1 else self.__dataset[self.__yName]
        self.__fit = self.__fit.fit(self.__dataset[self.__xName], y)
        self.params = list(self.__fit.coef_)
        intercept = float(self.__fit.intercept_)
        self.params.append(intercept)


class TestOLS(unittest.TestCase):
    """Test OLS
    The result data from two methods are the same.
    """

    def test_simulated(self):
        """Use the simulated data to test the models"""
        df = pd.read_csv('data/simulated_data.csv')
        ols_sm = OLS_SM(df, ['x1', 'x2'], ['y'])
        ols_sklearn = OLS_SKLEARN(df, ['x1', 'x2'], ['y'])
        np.testing.assert_allclose(ols_sm.params, ols_sklearn.params)

    def test_pm25(self):
        """Use the pm25 data to test the models"""
        df = pd.read_csv('data/pm25_data.csv')
        ols_sm = OLS_SM(df, ['dem', 'w10', 'd10', 't2m',
                        'aod_sat', 'tp'], ['PM2_5'])
        ols_sklearn = OLS_SKLEARN(
            df, ['dem', 'w10', 'd10', 't2m', 'aod_sat', 'tp'], ['PM2_5'])
        np.testing.assert_allclose(ols_sm.params, ols_sklearn.params)

    def test_sio3(self):
        """Use the sio3 data to test the models"""
        df = pd.read_csv('data/demo_data_gtnnwr.csv')
        ols_sm = OLS_SM(df, ['refl_b01', 'refl_b02', 'refl_b03',
                        'refl_b04', 'refl_b05', 'refl_b07'], ['SiO3'])
        ols_sklearn = OLS_SKLEARN(
            df, ['refl_b01', 'refl_b02', 'refl_b03', 'refl_b04', 'refl_b05', 'refl_b07'], ['SiO3'])
        np.testing.assert_allclose(ols_sm.params, ols_sklearn.params)
    
    def test_co2_01(self):
        """Use the co2 data to test the models"""
        df = pd.read_csv('data/co2_gnnwr.csv')
        ols_sm = OLS_SM(df, ['Chl', 'Temp', 'Salt'], ['fCO2'])
        ols_sklearn = OLS_SKLEARN(df, ['Chl', 'Temp', 'Salt'], ['fCO2'])
        np.testing.assert_allclose(ols_sm.params, ols_sklearn.params)
    
    def test_co2_02(self):
        """Use the co2 data to test the models"""
        df = pd.read_csv('data/co2_gtnnwr.csv')
        ols_sm = OLS_SM(df, ['Chl', 'Temp', 'Salt','pressure','windspeed'], ['pCO2'])
        ols_sklearn = OLS_SKLEARN(df, ['Chl', 'Temp', 'Salt','pressure','windspeed'], ['pCO2'])
        np.testing.assert_allclose(ols_sm.params, ols_sklearn.params)


if (__name__ == "__main__"):
    unittest.main()
