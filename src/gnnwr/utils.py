import math

import statsmodels.api as sm
import torch


class OLS():
    def __init__(self, dataset, xName: list, yName: str):  # xName为自变量字段名，yName为因变量字段名（需为列表）
        self.__dataset = dataset
        self.__xName = xName
        self.__yName = yName
        self.__formula = yName[0] + '~' + '+'.join(xName)
        self.__fit = sm.formula.ols(self.__formula, dataset).fit()
        self.params = list(self.__fit.params.to_dict().values())
        intercept = self.__fit.params[0]
        self.params = self.params[1:]
        self.params.append(intercept)
        #print(self.params)


class DIAGNOSIS:
    # TODO 更多诊断方法
    def __init__(self, weight, x_data, y_data, y_pred):
        self.__weight = weight
        self.__x_data = x_data
        self.__y_data = y_data
        self.__y_pred = y_pred
        self.__n = len(y_data)
        self.__k = len(x_data[0])

        self.__residual = y_data - y_pred
        self.__ssr = torch.sum((y_pred - torch.mean(y_data)) ** 2)

        self.__hat_com = torch.mm(torch.linalg.inv(
            torch.mm(self.__x_data.mT, self.__x_data)), self.__x_data.mT)
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
        
        return self.__hat

    def F1_GNN(self):  # 因变量的空间非平稳性F1检验
        k1 = self.__n - 2 * torch.trace(self.__hat) + \
             torch.trace(torch.dot(self.__hat.mT, self.__hat))
        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum(
            (torch.mean(self.__y_data) - torch.dot(self.__ols_hat, self.__y_data)) ** 2)
        return self.__ssr / k1 / (rss_olr / k2)

    def AIC(self):
        return self.__n * (math.log(self.__ssr / self.__n * 2 * math.pi, math.e)) + self.__n + self.__k
    def AICc(self):
        return self.__n * (math.log(self.__ssr / self.__n * 2 * math.pi, math.e) + (self.__n + self.__S) / (
                self.__n - self.__S - 2))

    def R2(self):
        return 1 - torch.sum(self.__residual ** 2) / torch.sum((self.__y_data - torch.mean(self.__y_data)) ** 2)

    def Adjust_R2(self):
        return 1 - (1 - self.R2()) * (self.__n - 1) / (self.__n - self.__k - 1)

    def RMSE(self):
        return torch.sqrt(torch.sum(self.__residual ** 2) / self.__n)
