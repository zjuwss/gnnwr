import math
import statsmodels.api as sm
import torch


class OLS():
    def __init__(self, dataset, xName: list, yName: list):
        """
        OLS is the class to calculate the OLR weights of data.Get the weight by `object.params`.

        :param dataset: Input data
        :param xName: the independent variables' column
        :param yName:the dependent variable's column
        """
        self.__dataset = dataset
        self.__xName = xName
        self.__yName = yName
        self.__formula = yName[0] + '~' + '+'.join(xName)
        self.__fit = sm.formula.ols(self.__formula, dataset).fit()
        self.params = list(self.__fit.params.to_dict().values())
        intercept = self.__fit.params[0]
        self.params = self.params[1:]
        self.params.append(intercept)


class DIAGNOSIS:
    # TODO 更多诊断方法
    def __init__(self, weight, x_data, y_data, y_pred):
        """
        Diagnosis is the class to calculate the diagnoses of GNNWR/GTNNWR.

        :param weight: output of the neural network
        :param x_data: the independent variables
        :param y_data: the dependent variables
        :param y_pred: output of the GNNWR/GTNNWR
        """
        self.__weight = weight
        self.__x_data = x_data
        self.__y_data = y_data
        self.__y_pred = y_pred
        self.__n = len(y_data)
        self.__k = len(x_data[0])

        self.__residual = y_data - y_pred
        self.__ssr = torch.sum((y_pred - torch.mean(y_data)) ** 2)

        self.__hat_com = torch.mm(torch.linalg.inv(
            torch.mm(self.__x_data.transpose(-2,-1), self.__x_data)), self.__x_data.transpose(-2,-1))
        self.__ols_hat = torch.mm(self.__x_data, self.__hat_com)
        x_data_tile = x_data.repeat(self.__n, 1)
        x_data_tile = x_data_tile.view(self.__n, self.__n, -1)
        x_data_tile_t = x_data_tile.transpose(1, 2)
        gtweight_3d = torch.diag_embed(self.__weight)
        hatS_temp = torch.matmul(gtweight_3d,
                                 torch.matmul(torch.inverse(torch.matmul(x_data_tile_t, x_data_tile)), x_data_tile_t))
        hatS = torch.matmul(x_data.view(-1, 1, x_data.size(1)), hatS_temp)
        hatS = hatS.view(-1, self.__n)
        self.__hat = hatS
        self.__S = torch.trace(self.__hat)

    def hat(self):
        """
        :return: hat matrix
        """
        return self.__hat

    def F1_GNN(self):
        """
        :return: F1-test
        """
        k1 = self.__n - 2 * torch.trace(self.__hat) + \
             torch.trace(torch.mm(self.__hat.transpose(-2,-1), self.__hat))
        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum(
            (torch.mean(self.__y_data) - torch.mm(self.__ols_hat, self.__y_data)) ** 2)
        return self.__ssr / k1 / (rss_olr / k2)

    def AIC(self):
        """
        :return: AIC
        """
        return self.__n * (math.log(self.__ssr / self.__n * 2 * math.pi, math.e)) + self.__n + self.__k
    def AICc(self):
        """

        :return: AICc
        """
        return self.__n * (math.log(self.__ssr / self.__n * 2 * math.pi, math.e) + (self.__n + self.__S) / (
                self.__n - self.__S - 2))

    def R2(self):
        """

        :return: R2 of the result
        """
        return 1 - torch.sum(self.__residual ** 2) / torch.sum((self.__y_data - torch.mean(self.__y_data)) ** 2)

    def Adjust_R2(self):
        """

        :return: Adjust R2 of the result
        """
        return 1 - (1 - self.R2()) * (self.__n - 1) / (self.__n - self.__k - 1)

    def RMSE(self):
        """

        :return: RMSE of the result
        """
        return torch.sqrt(torch.sum(self.__residual ** 2) / self.__n)
