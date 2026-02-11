import math
from sklearn.linear_model import LinearRegression
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
        self.__fit = LinearRegression(fit_intercept=True)
        y = self.__dataset[self.__yName[0]] if len(
            self.__yName) == 1 else self.__dataset[self.__yName]
        self.__fit = self.__fit.fit(self.__dataset[self.__xName], y)
        self.params = list(self.__fit.coef_)
        intercept = float(self.__fit.intercept_)
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
    :param lite: If True, skip all Hat-dependent diagnostics (only R²/RMSE).
           If False or None (default), compute all diagnostics efficiently
           in O(n·k²) without forming the n×n Hat matrix.
    """

    def __init__(self, weight, x_data, y_data, y_pred, lite=None):
        self._device = torch.device('cuda') if weight.is_cuda else torch.device('cpu')

        self.__weight = weight.clone()
        self.__x_data = x_data.clone()
        self.__y_data = y_data.clone()
        self.__y_pred = y_pred.clone()

        self.__n = len(self.__y_data)
        self.__k = len(self.__x_data[0])

        self.__residual = self.__y_data - self.__y_pred
        self.__ssr = torch.sum((self.__y_pred - self.__y_data) ** 2)

        self._lite = (lite is True)
        self._mode = 'skip' if self._lite else 'full'
        self.f3_dict = None
        self.f3_dict_2 = None

        if not self._lite:
            self._compute_efficient()

    def _compute_efficient(self):
        """Compute all diagnostic quantities in O(n·k²) without n×n matrices.

        Key insight: the Hat matrix S[i,j] = (x_i ⊙ w_i)^T A x_j where
        A = (X^T X)^{-1}.  All diagnostics (tr(S), tr(S^T S), Sy, S^T y, etc.)
        can be derived from the k×k matrix A and O(n·k) vector operations,
        avoiding the O(n²) memory cost of forming S explicitly.
        """
        X = self.__x_data    # (n, k)
        W = self.__weight     # (n, k)
        y = self.__y_data     # (n, 1)

        # A = (X^T X)^{-1}, shape (k, k)
        XtX = torch.mm(X.t(), X)
        A = torch.linalg.inv(XtX)

        # hat_com = A @ X^T, shape (k, n) — stored for F3 and hat()
        self.__hat_com = torch.mm(A, X.t())

        # XW = X ⊙ W, shape (n, k)
        XW = X * W

        # tr(S) = Σ_i (XW[i] · hat_com[:, i])
        self.__S = torch.sum(XW * self.__hat_com.t())

        # tr(S^T S) = sum(B ⊙ (XtX @ B)) where B = A @ XW^T
        B = torch.mm(A, XW.t())  # (k, n)
        self.__trStS = torch.sum(B * torch.mm(XtX, B))

        # Precompute v = X^T y, Av = A v
        v = torch.mm(X.t(), y)   # (k, 1)
        Av = torch.mm(A, v)      # (k, 1)

        # H_ols @ y = X @ Av
        self.__Hy = torch.mm(X, Av)       # (n, 1)

        # S @ y = XW @ Av
        self.__Sy = torch.mm(XW, Av)      # (n, 1)

        # S^T @ y = X @ A @ (XW^T @ y)
        u = torch.mm(XW.t(), y)           # (k, 1)
        self.__Sty = torch.mm(X, torch.mm(A, u))  # (n, 1)

    def _require_diagnostics(self, method_name):
        """Ensure diagnostics have been computed (not in lite mode)."""
        if self._lite:
            raise RuntimeError(
                f"{method_name}() is disabled in lite=True mode. "
                f"Use lite=False or lite=None to enable all diagnostics."
            )

    def hat(self):
        """
        :return: hat matrix (n×n, computed on demand — use with caution for large n)
        """
        self._require_diagnostics("hat")
        if self.__n > 10000:
            warnings.warn(
                f"Computing full n×n Hat matrix for n={self.__n} "
                f"requires ~{self.__n**2 * 4 / 1e9:.1f}GB memory.")
        XW = self.__x_data * self.__weight
        return torch.mm(XW, self.__hat_com)

    def F1_Global(self):
        """
        :return: F1-test
        """
        self._require_diagnostics("F1_Global")
        k1 = self.__n - 2 * self.__S + self.__trStS
        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum((self.__y_data - self.__Hy) ** 2)
        F_value = self.__ssr / k1 / (rss_olr / k2)
        return F_value

    def F2_Global(self):
        """
        :return: F2-test
        """
        self._require_diagnostics("F2_Global")
        # tr(A_mat) = -k + 2*tr(S) - tr(S^T S)
        v1 = -self.__k + 2 * self.__S - self.__trStS
        # y^T A_mat y = -y^T Hy + y^T Sy + y^T S^T y - ||Sy||²
        yHy = torch.mm(self.__y_data.t(), self.__Hy)
        ySy = torch.mm(self.__y_data.t(), self.__Sy)
        ySty = torch.mm(self.__y_data.t(), self.__Sty)
        SySy = torch.mm(self.__Sy.t(), self.__Sy)
        DSS = -yHy + ySy + ySty - SySy

        k2 = self.__n - self.__k - 1
        rss_olr = torch.sum(
            (torch.mean(self.__y_data) - self.__Hy) ** 2)

        return DSS / v1 / (rss_olr / k2)

    def F3_Local(self):
        """
        :return: F3-test of each variable
        """
        self._require_diagnostics("F3_Local")
        self.f3_dict = {}
        self.f3_dict_2 = {}
        y_flat = self.__y_data.squeeze()  # (n,)
        n = self.__n

        for l in range(self.__k):
            w_l = self.__weight[:, l]      # (n,)
            h_l = self.__hat_com[l, :]     # (n,)

            # hatB is rank-1: hatB[a,b] = w_l[a] * h_l[b]
            # L = hatB^T @ M @ hatB where M = I - J (all-ones matrix)
            # L = c * (h_l ⊗ h_l) where c = ||w_l||² - sum(w_l)²
            sum_wl = w_l.sum()
            c = torch.dot(w_l, w_l) - sum_wl * sum_wl

            h_dot_y = torch.dot(h_l, y_flat)
            h_norm_sq = torch.dot(h_l, h_l)

            # vk2 = (c/n) * (h·y)²,  trace_L/n = (c/n) * ||h||²
            trace_L_n = c / n * h_norm_sq
            vk2 = c / n * h_dot_y ** 2
            f3 = torch.squeeze(vk2 / trace_L_n / (self.__ssr / n))
            self.f3_dict['f3_param_' + str(l)] = f3

            # Second variant: bk = w_l * (h·y), centered
            mean_wl = sum_wl / n
            vk2_2 = h_dot_y ** 2 / n * torch.sum((w_l - mean_wl) ** 2)
            f3_2 = torch.squeeze(vk2_2 / trace_L_n / (self.__ssr / n))
            self.f3_dict_2['f3_param_' + str(l)] = f3_2

        return self.f3_dict, self.f3_dict_2

    def AIC(self):
        """
        :return: AIC
        """
        self._require_diagnostics("AIC")
        return self.__n * (math.log(self.__ssr / self.__n * 2 * math.pi, math.e)) + self.__n + self.__S

    def AICc(self):
        """

        :return: AICc
        """
        self._require_diagnostics("AICc")
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
