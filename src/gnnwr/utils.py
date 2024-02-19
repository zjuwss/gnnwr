import math
import statsmodels.api as sm
import pandas as pd
import torch
import warnings
import folium
from folium.plugins import HeatMap
import branca


class OLS:
    """
    `OLS` is the class to calculate the OLR coefficients of data.Get the coefficient by `object.params`.

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
        intercept = self.__fit.params[0]
        self.params = self.params[1:]
        self.params.append(intercept)


class DIAGNOSIS:
    """
    `DIAGNOSIS` is the class to calculate the diagnoses of the result of GNNWR/GTNNWR.
    These diagnoses include F1-test, F2-test, F3-test, AIC, AICc, R2, Adjust_R2, RMSE (Root Mean Square Error).
    The explanation of these diagnoses can be found in the paper 
    `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`.
    :param weight: output of the neural network
    :param x_data: the independent variables
    :param y_data: the dependent variables
    :param y_pred: output of the GNNWR/GTNNWR
    """

    def __init__(self, weight, x_data, y_data, y_pred):
        self.__weight = weight.clone().to('cpu')
        self.__x_data = x_data.clone().to('cpu')
        self.__y_data = y_data.clone().to('cpu')
        self.__y_pred = y_pred.clone().to('cpu')

        self.__n = len(self.__y_data)
        self.__k = len(self.__x_data[0])

        self.__residual = self.__y_data - self.__y_pred
        self.__ssr = torch.sum((self.__y_pred - self.__y_data) ** 2) # sum of squared residuals

        self.__hat_com = torch.mm(torch.linalg.inv(
            torch.mm(self.__x_data.transpose(-2, -1), self.__x_data)), self.__x_data.transpose(-2, -1))
        self.__ols_hat = torch.mm(self.__x_data, self.__hat_com)
        x_data_tile = self.__x_data.repeat(self.__n, 1)
        x_data_tile = x_data_tile.view(self.__n, self.__n, -1)
        x_data_tile_t = x_data_tile.transpose(1, 2)
        gtweight_3d = torch.diag_embed(self.__weight)

        hatS_temp = torch.matmul(gtweight_3d,
                                 torch.matmul(torch.inverse(torch.matmul(x_data_tile_t, x_data_tile)), x_data_tile_t))
        self.__hat_temp = hatS_temp
        hatS = torch.matmul(self.__x_data.view(-1, 1, self.__x_data.size(1)), hatS_temp)
        hatS = hatS.view(-1, self.__n)
        self.__hat = hatS
        self.__S = torch.trace(self.__hat)
        self.f3_dict = None
        self.f3_dict_2 = None
    def hat(self):
        """
        :return: hat matrix
        """
        return self.__hat

    def F1_Global(self):
        """
        :return: F1-test
        """
        k1 = self.__n - 2 * torch.trace(self.__hat) + \
             torch.trace(torch.mm(self.__hat.transpose(-2, -1), self.__hat))

        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum(
            (self.__y_data - torch.mm(self.__ols_hat, self.__y_data)) ** 2)
        F_value = self.__ssr / k1 / (rss_olr / k2)
        # p_value = f.sf(F_value, k1, k2)
        return F_value

    def F2_Global(self):
        """
        :return: F2-test
        """
        # A = (I - H) - (I - S)^T*(I - S)
        A = (torch.eye(self.__n) - self.__ols_hat) - torch.mm(
            (torch.eye(self.__n) - self.__hat).transpose(-2, -1),
            (torch.eye(self.__n) - self.__hat))
        v1 = torch.trace(A)
        # DSS = y^T*A*y
        DSS = torch.mm(self.__y_data.transpose(-2, -1), torch.mm(A, self.__y_data))
        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum(
            (torch.mean(self.__y_data) - torch.mm(self.__ols_hat, self.__y_data)) ** 2)

        return DSS / v1 / (rss_olr / k2)

    def F3_Local(self):
        """
        :return: F3-test of each variable
        """

        ek_dict = {}
        self.f3_dict = {}
        self.f3_dict_2 = {}
        for i in range(self.__x_data.size(1)):
            ek_zeros = torch.zeros([self.__x_data.size(1)])
            ek_zeros[i] = 1
            ek_dict['ek' + str(i)] = torch.reshape(torch.reshape(torch.tile(ek_zeros.clone().detach(), [self.__n]),
                                                                 [self.__n, -1]),
                                                   [-1, 1, self.__x_data.size(1)])
            hatB = torch.matmul(ek_dict['ek' + str(i)], self.__hat_temp)
            hatB = torch.reshape(hatB, [-1, self.__n])

            J_n = torch.ones([self.__n, self.__n]) / self.__n
            L = torch.matmul(hatB.transpose(-2, -1), torch.matmul(torch.eye(self.__n) - J_n, hatB))

            vk2 = 1 / self.__n * torch.matmul(self.__y_data.transpose(-2, -1), torch.matmul(L, self.__y_data))
            trace_L = torch.trace(1 / self.__n * L)
            f3 = torch.squeeze(vk2 / trace_L / (self.__ssr / self.__n))
            self.f3_dict['f3_param_' + str(i)] = f3

            bk = torch.matmul(hatB, self.__y_data)
            vk2_2 = 1 / self.__n * torch.sum((bk - torch.mean(bk)) ** 2)
            f3_2 = torch.squeeze(vk2_2 / trace_L / (self.__ssr / self.__n))
            self.f3_dict_2['f3_param_' + str(i)] = f3_2
        return self.f3_dict, self.f3_dict_2

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


class Visualize:
    """
    `Visualize` is the class to visualize the data and the result of GNNWR/GTNNWR.
    It based on the `folium` package and use GaoDe map as the background. And it can display the dataset, the coefficients heatmap, and the dot map,
    which helps to understand the spatial distribution of the data and the result of GNNWR/GTNNWR better.
    
    :param data: the input data
    :param lon_lat_columns: the columns of longitude and latitude
    :param zoom: the zoom of the map
    """
    def __init__(self, data, lon_lat_columns=None, zoom=4):
        self.__raw_data = data
        self.__tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=en&size=1&scl=1&style=7'
        self.__zoom = zoom
        if hasattr(self.__raw_data, '_use_gpu'):
            self._train_dataset = self.__raw_data._train_dataset.dataframe
            self._valid_dataset = self.__raw_data._valid_dataset.dataframe
            self._test_dataset = self.__raw_data._test_dataset.dataframe
            self._result_data = self.__raw_data.result_data
            self._all_data = pd.concat([self._train_dataset, self._valid_dataset, self._test_dataset])
            if lon_lat_columns is None:
                warnings.warn("lon_lat columns are not given. Using the spatial columns in dataset")
                self._spatial_column = self._train_dataset.spatial_column
                self.__center_lon = self._all_data[self._spatial_column[0]].mean()
                self.__center_lat = self._all_data[self._spatial_column[1]].mean()
                self.__lon_column = self._spatial_column[0]
                self.__lat_column = self._spatial_column[1]
            else:
                self._spatial_column = lon_lat_columns
                self.__center_lon = self._all_data[self._spatial_column[0]].mean()
                self.__center_lat = self._all_data[self._spatial_column[1]].mean()
                self.__lon_column = self._spatial_column[0]
                self.__lat_column = self._spatial_column[1]
            self._x_column = data._train_dataset.x_column
            self._y_column = data._train_dataset.y_column
            self.__map = folium.Map(location=[self.__center_lat, self.__center_lon], zoom_start=zoom,
                                    tiles=self.__tiles, attr="高德")
        else:
            raise ValueError("given data is not instance of GNNWR")

    def display_dataset(self, name="all", y_column=None, colors=None, steps=20, vmin=None, vmax=None):
        """
        Display the dataset on the map, including the train, valid, test dataset.
        
        :param name: the name of the dataset, including 'all', 'train', 'valid', 'test'
        :param y_column: the column of the displayed variable
        :param colors: the list of colors, if not given, the default color is used
        :param steps: the steps of the colors
        
        """
        if colors is None:
            colors = []
        if y_column is None:
            warnings.warn("y_column is not given. Using the first y_column in dataset")
            y_column = self._y_column[0]
        if name == 'all':
            dst = self._all_data
        elif name == 'train':
            dst = self._train_dataset
        elif name == 'valid':
            dst = self._valid_dataset
        elif name == 'test':
            dst = self._test_dataset
        else:
            raise ValueError("name is not included in 'all','train','valid','test'")
        dst_min = dst[y_column].min() if vmin == None else vmin
        dst_max = dst[y_column].max() if vmax == None else vmax
        res = folium.Map(location=[self.__center_lat, self.__center_lon], zoom_start=self.__zoom, tiles=self.__tiles,
                         attr="高德")
        if len(colors) <= 0:
            colormap = branca.colormap.linear.YlOrRd_09.scale(dst_min, dst_max).to_step(steps)
        else:
            colormap = branca.colormap.LinearColormap(colors=colors, vmin=dst_min, vmax=dst_max).to_step(steps)
        for idx, row in dst.iterrows():
            folium.CircleMarker(location=(row[self.__lat_column], row[self.__lon_column]), radius=7,
                                color=colormap.rgb_hex_str(row[y_column]), fill=True, fill_opacity=1,
                                popup="""
            longitude:{}
            latitude:{}
            {}:{}
            """.format(row[self.__lon_column], row[self.__lat_column], y_column, row[y_column])
                                ).add_to(res)
        res.add_child(colormap)
        return res

    def coefs_heatmap(self, data_column, colors=None, steps=20, vmin=None, vmax=None):
        """
        Display the heatmap of the coefficients of the result of GNNWR/GTNNWR.

        :param data_column: the column of the displayed variable
        :param colors: the list of colors, if not given, the default color is used
        :param steps: the steps of the colors
        :param vmin: the minimum value of the displayed variable, if not given, the minimum value of the variable is used
        :param vmax: the maximum value of the displayed variable, if not given, the maximum value of the variable is used
        """
        if colors is None:
            colors = []
        res = folium.Map(location=[self.__center_lat, self.__center_lon], zoom_start=self.__zoom, tiles=self.__tiles,
                         attr="高德")
        dst = self._result_data
        dst_min = dst[data_column].min() if vmin is None else vmin
        dst_max = dst[data_column].max() if vmax is None else vmax
        data = [[row[self.__lat_column], row[self.__lon_column], row[data_column]] for index, row in dst.iterrows()]
        if len(colors) <= 0:
            colormap = branca.colormap.linear.YlOrRd_09.scale(dst_min, dst_max).to_step(steps)
        else:
            colormap = branca.colormap.LinearColormap(colors=colors, vmin=dst_min, vmax=dst_max).to_step(steps)
        gradient_map = dict()
        for i in range(steps):
            gradient_map[i / steps] = colormap.rgb_hex_str(i / steps)
        colormap.add_to(res)
        HeatMap(data=data, gradient=gradient_map, radius=10).add_to(res)
        return res

    def dot_map(self, data, lon_column, lat_column, y_column, zoom=4, colors=None, steps=20, vmin=None, vmax=None):
        """
        Display the data by dot map, the color of the dot represents the value of the variable.
        
        :param data: the input data
        :param lon_column: the column of longitude
        :param lat_column: the column of latitude
        :param y_column: the column of the displayed variable
        :param zoom: the zoom of the map
        :param colors: the list of colors, if not given, the default color is used
        :param steps: the steps of the colors
        :param vmin: the minimum value of the displayed variable, if not given, the minimum value of the variable is used
        :param vmax: the maximum value of the displayed variable, if not given, the maximum value of the variable is used
        """
        if colors is None:
            colors = []
        center_lon = data[lon_column].mean()
        center_lat = data[lat_column].mean()
        dst_min = data[y_column].min() if vmin is None else vmin
        dst_max = data[y_column].max() if vmax is None else vmax
        res = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=self.__tiles, attr="高德")
        if len(colors) <= 0:
            colormap = branca.colormap.linear.YlOrRd_09.scale(dst_min, dst_max).to_step(steps)
        else:
            colormap = branca.colormap.LinearColormap(colors=colors, vmin=dst_min, vmax=dst_max).to_step(steps)
        for idx, row in data.iterrows():
            folium.CircleMarker(location=(row[lat_column], row[lon_column]), radius=7,
                                color=colormap.rgb_hex_str(row[y_column]), fill=True, fill_opacity=1,
                                popup="""
            longitude:{}
            latitude:{}
            {}:{}
            """.format(row[lon_column], row[lat_column], y_column, row[y_column])
                                ).add_to(res)
        colormap.add_to(res)
        return res
