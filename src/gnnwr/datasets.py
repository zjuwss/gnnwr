import json
import os

import numpy as np
import pandas
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
from scipy.spatial import distance

r"""
The package of `datasets` includes the following functions:
    1. init_dataset: initialize the dataset for training, validation and testing
    2. init_dataset_cv: initialize the dataset for cross-validation
    3. init_predict_dataset: initialize the dataset for prediction
    4. BasicDistance: calculate the distance matrix of spatial/spatio-temporal data
    5. ManhattanDistance: calculate the Manhattan distance matrix of spatial/spatio-temporal data
and the following classes:
    1. baseDataset: the base class of dataset
    2. predictDataset: the class of dataset for prediction
the purpose of this package is to provide the basic functions of pre-processing data and calculating distance matrix
to facilitate the use of the model.
"""


class baseDataset(Dataset):
    r"""
    baseDataset is the base class of dataset, which is used to store the data and other information.
    it also provides the function of data scaling, data saving and data loading.

    Parameters
    ----------
    data: pandas.DataFrame
        dataframe
    x_column: list
        independent variable column name
    y_column: list
        dependent variable column name
    id_column: str
        id column name
    is_need_STNN: bool
        whether need STNN(default: ``False``)
        | if ``True``, the dataset will be used to train the Model with STNN and SPNN
        | and the GTNNWR Model will use the STNN and SPNN to calculate the distance matrix
        | if ``False``, the dataset will not be used to train the Model with STNN and SPNN
    """

    def __init__(self, data=None, x_column: list = None, y_column: list = None, id_column=None, is_need_STNN=False):

        self.dataframe = data
        self.x = x_column
        self.y = y_column
        self.id = id_column
        if data is None:
            self.x_data = None
            self.datasize = -1
            self.coefsize = -1
            self.y_data = None
            self.id_data = None
        else:
            self.x_data = data[x_column].astype(np.float32).values  # x_data is independent variables data
            self.datasize = self.x_data.shape[0]  # datasize is the number of samples
            self.coefsize = len(x_column) + 1  # coefsize is the number of coefficients
            self.y_data = data[y_column].astype(np.float32).values  # y_data is dependent variables data
            if id_column is not None:
                self.id_data = data[id_column].astype(np.int64).values
            else:
                raise ValueError("id_column is None")
        self.is_need_STNN = is_need_STNN
        self.scale_fn = None  # scale function
        self.x_scale_info = None  # scale information of x_data
        self.y_scale_info = None  # scale information of y_data
        self.distances = None  # distances is the distance matrix of spatial/spatio-temporal data
        self.temporal = None  # temporal is the temporal distance matrix of spatio-temporal data
        self.distances_scale_params = None  # scale parameters of distances
        self.simple_distance = True
        self.scaledDataframe = None
        self.batch_size = None
        self.shuffle = None
        self.distances_scale_param = None

    def __len__(self):
        """
        :return: the number of samples
        """
        return len(self.y_data)

    def __getitem__(self, index):
        """
        :param index: the index of sample
        :return: the index-th distance matrix and the index-th sample
        """
        if self.is_need_STNN:
            return torch.cat((torch.tensor(self.distances[index], dtype=torch.float),
                              torch.tensor(self.temporal[index], dtype=torch.float)), dim=-1), \
                torch.tensor(self.x_data[index], dtype=torch.float), \
                torch.tensor(self.y_data[index], dtype=torch.float), \
                torch.tensor(self.id_data[index], dtype=torch.float)
        return torch.tensor(self.distances[index], dtype=torch.float), torch.tensor(self.x_data[index],
                                                                                    dtype=torch.float), torch.tensor(
            self.y_data[index], dtype=torch.float), torch.tensor(self.id_data[index], dtype=torch.float)

    def scale(self, scale_fn=None, scale_params=None):
        """
        scale the data by MinMaxScaler or StandardScaler
        | the scale function will scale the independent variable data and add a column of 1 to the data

        Parameters
        ----------
        scale_fn: str
            scale function name
            | if ``minmax_scale``, use MinMaxScaler
            | if ``standard_scale``, use StandardScaler
        scale_params: list
            scaler with scale parameters
            | if ``minmax_scale``, scale_params is a list of MinMaxScaler
            | if ``standard_scale``, scale_params is a list of StandardScaler

        """
        if scale_fn == "minmax_scale":
            self.scale_fn = "minmax_scale"
            x_scale_params = scale_params[0]
            y_scale_params = scale_params[1]
            self.x_scale_info = {"min": x_scale_params.data_min_, "max": x_scale_params.data_max_}
            self.x_data = x_scale_params.transform(pd.DataFrame(self.x_data, columns=self.x))
            self.y_scale_info = {"min": y_scale_params.data_min_, "max": y_scale_params.data_max_}
        elif scale_fn == "standard_scale":
            self.scale_fn = "standard_scale"
            x_scale_params = scale_params[0]
            y_scale_params = scale_params[1]
            self.x_scale_info = {"mean": x_scale_params.mean_, "var": x_scale_params.var_}
            self.x_data = x_scale_params.transform(pd.DataFrame(self.x_data, columns=self.x))
            self.y_scale_info = {"mean": y_scale_params.mean_, "var": y_scale_params.var_}

        self.getScaledDataframe()

        self.x_data = np.concatenate((self.x_data, np.ones(
            (self.datasize, 1))), axis=1)

    def scale2(self, scale_fn, scale_params):
        """
        scale the data with the scale function and scale parameters

        Parameters
        ----------
        scale_fn: str
            scale function name
            | if ``minmax_scale``, use MinMaxScaler
            | if ``standard_scale``, use StandardScaler
        scale_params: list
            scaler with scale parameters
            | if ``minmax_scale``, scale_params is a list of dict with ``min`` and ``max``
            | if ``standard_scale``, scale_params is a list of dict with ``mean`` and ``var``
        """
        if scale_fn == "minmax_scale":
            self.scale_fn = "minmax_scale"
            x_scale_params = scale_params[0]
            y_scale_params = scale_params[1]
            self.x_data = (self.x_data - x_scale_params["min"]) / (x_scale_params["max"] - x_scale_params["min"])
        elif scale_fn == "standard_scale":
            self.scale_fn = "standard_scale"
            x_scale_params = scale_params[0]
            y_scale_params = scale_params[1]
            self.x_data = (self.x_data - x_scale_params['mean']) / np.sqrt(x_scale_params["var"])

        self.getScaledDataframe()

        self.x_data = np.concatenate((self.x_data, np.ones(
            (self.datasize, 1))), axis=1)

    def getScaledDataframe(self):
        """
        get the scaled dataframe and save it in ``scaledDataframe``
        """
        columns = np.concatenate((self.x, self.y), axis=0)
        scaledData = np.concatenate((self.x_data, self.y_data), axis=1)
        self.scaledDataframe = pd.DataFrame(scaledData, columns=columns)

    def rescale(self, x):
        """
        rescale the data with the scale function and scale parameters

        Parameters
        ----------
        x: numpy.ndarray
            independent variable data
        y: numpy.ndarray
            dependent variable data

        Returns
        -------
        x: numpy.ndarray
            rescaled independent variable data
        y: numpy.ndarray
            rescaled dependent variable data
        """
        if self.scale_fn == "minmax_scale":
            x = np.multiply(x, self.x_scale_info["max"] - self.x_scale_info["min"]) + self.x_scale_info["min"]
        elif self.scale_fn == "standard_scale":
            x = np.multiply(x, np.sqrt(self.x_scale_info["var"])) + self.x_scale_info["mean"]
        else:
            raise ValueError("invalid process_fn")
        return x


    def save(self, dirname):
        """
        save the dataset

        :param dirname: save directory
        """
        if os.path.exists(dirname):
            raise ValueError("dir is already exists")
        if self.dataframe is None:
            raise ValueError("dataframe is None")
        os.makedirs(dirname)
        x_scale_info = {}
        y_scale_info = {}
        for key, value in self.x_scale_info.items():
            x_scale_info[key] = value.tolist()
        for key, value in self.y_scale_info.items():
            y_scale_info[key] = value.tolist()
        with open(os.path.join(dirname, "dataset_info.json"), "w") as f:
            distance_scale_info = {}
            for key in self.distances_scale_param.keys():
                distance_scale_info[key] = self.distances_scale_param[key].tolist()
            json.dump({"x": self.x,
                       "y": self.y,
                       "id": self.id,
                       "batch_size": self.batch_size,
                       "shuffle": self.shuffle,
                       "is_need_STNN": self.is_need_STNN,
                       "scale_fn": self.scale_fn,
                       "x_scale_info": json.dumps(x_scale_info),
                       "y_scale_info": json.dumps(y_scale_info),
                       "distance_scale_info": json.dumps(distance_scale_info),
                       'simple_distance': self.simple_distance
                       }, f)
        # save the distance matrix
        np.save(os.path.join(dirname, "distances.npy"), self.distances)
        # save dataframe
        self.dataframe.to_csv(os.path.join(dirname, "dataframe.csv"), index=False)
        self.scaledDataframe.to_csv(os.path.join(dirname, "scaledDataframe.csv"), index=False)

    def read(self, dirname):
        """
        read the dataset by the directory

        :param dirname: read directory
        """
        if not os.path.exists(dirname):
            raise ValueError("dir is not exists")
        # read the information of dataset
        with open(os.path.join(dirname, "dataset_info.json"), "r") as f:
            dataset_info = json.load(f)
        self.x = dataset_info["x"]
        self.y = dataset_info["y"]
        self.id = dataset_info["id"]
        self.batch_size = dataset_info["batch_size"]
        self.shuffle = dataset_info["shuffle"]
        self.is_need_STNN = dataset_info["is_need_STNN"]
        self.scale_fn = dataset_info["scale_fn"]
        self.simple_distance = dataset_info["simple_distance"]
        self.x_scale_info = json.loads(dataset_info["x_scale_info"])
        self.y_scale_info = json.loads(dataset_info["y_scale_info"])
        self.distances_scale_param = json.loads(dataset_info["distance_scale_info"])
        x_scale_info = self.x_scale_info
        y_scale_info = self.y_scale_info
        for key, value in x_scale_info.items():
            x_scale_info[key] = np.array(value)
        for key, value in y_scale_info.items():
            y_scale_info[key] = np.array(value)
        # read the distance matrix
        self.distances = np.load(os.path.join(dirname, "distances.npy")).astype(np.float32)
        # read dataframe
        self.dataframe = pd.read_csv(os.path.join(dirname, "dataframe.csv"))
        self.x_data = self.dataframe[self.x].astype(np.float32).values
        self.datasize = self.x_data.shape[0]
        self.y_data = self.dataframe[self.y].astype(np.float32).values
        self.id_data = self.dataframe[self.id].astype(np.int64).values
        self.coefsize = len(self.x) + 1
        self.scale2(self.scale_fn, [self.x_scale_info, self.y_scale_info])


class predictDataset(Dataset):
    """
    Predict dataset is used to predict the dependent variable of the data.

    :param data: dataframe
    :param x_column: independent variable column name
    :param process_fn: process function name
    :param scale_info: process function parameters
    :param is_need_STNN: whether to need STNN
    """

    def __init__(self, data, x_column, process_fn="minmax_scale", scale_info=None, is_need_STNN=False):

        # data = data.astype(np.float32)
        if scale_info is None:
            scale_info = []
        self.dataframe = data
        self.x = x_column
        if data is None:
            self.x_data = None
            self.datasize = -1
            self.coefsize = -1
        else:
            self.x_data = data[x_column].astype(np.float32).values  # x_data is independent variables data
            self.datasize = self.x_data.shape[0]  # datasize is the number of samples
            self.coefsize = len(x_column) + 1  # coefsize is the number of coefficients
        self.is_need_STNN = is_need_STNN
        self.process_fn = process_fn
        if len(scale_info):
            self.scale_info_x = scale_info[0]  # scale information of x_data
            self.use_scale_info = True
        else:
            self.use_scale_info = False
        # 数据预处理
        if process_fn == "minmax_scale":
            self.scale_fn = "minmax_scale"
            # stander = MinMaxScaler()
            # self.x_data = stander.fit_transform(self.x_data)
            if self.use_scale_info:
                self.x_data = self.minmax_scaler(self.x_data, self.scale_info_x[0], self.scale_info_x[1])
            else:
                self.x_data = self.minmax_scaler(self.x_data)
        elif process_fn == "standard_scale":
            self.scale_fn = "standard_scale"
            # stander = StandardScaler()
            # self.x_data = stander.fit_transform(self.x_data)
            if self.use_scale_info:
                self.x_data = self.standard_scaler(self.x_data, self.scale_info_x[0], self.scale_info_x[1])
            else:
                self.x_data = self.standard_scaler(self.x_data)

        else:
            raise ValueError("invalid process_fn")

        self.x_data = np.concatenate((self.x_data, np.ones(
            (self.datasize, 1))), axis=1)

        self.distances = None
        self.temporal = None

    def __len__(self):
        """
        :return: the number of samples
        """
        return len(self.x_data)

    def __getitem__(self, index):
        """
        :param index: sample index
        :return: distance matrix and independent variable data and dependent variable data
        """
        if self.is_need_STNN:
            return torch.cat((torch.tensor(self.distances[index], dtype=torch.float),
                              torch.tensor(self.temporal[index], dtype=torch.float)), dim=-1), torch.tensor(
                self.x_data[index], dtype=torch.float)
        return torch.tensor(self.distances[index], dtype=torch.float), torch.tensor(self.x_data[index],
                                                                                    dtype=torch.float)

    def rescale(self, x):
        """
        rescale the attribute data

        :param x: Input attribute data
        :return: rescaled attribute data
        """
        if self.scale_fn == "minmax_scale":
            x = x * (self.scale_info_x[1] - self.scale_info_x[0]) + self.scale_info_x[0]
        elif self.scale_fn == "standard_scale":
            x = x * np.sqrt(self.scale_info_x[1]) + self.scale_info_x[0]
        else:
            raise ValueError("invalid process_fn")

        return x

    def minmax_scaler(self, x, min=None, max=None):
        """
        function of minmax scaler

        :param x: Input attribute data
        :param min: minimum value of each attribute
        :param max: maximum value of each attribute
        :return: Output attribute data
        """
        if max is None:
            max = []
        if min is None:
            min = []
        if len(min) == 0:
            x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        else:
            x = (x - min) / (max - min)
        return x

    def standard_scaler(self, x, mean=None, std=None):
        """
        function of standard scaler

        :param x: Input attribute data
        :param mean: mean value of each attribute
        :param std: standard deviation of each attribute
        :return: Output attribute data
        """
        if std is None:
            std = []
        if mean is None:
            mean = []
        if len(mean) == 0:
            x = (x - x.mean(axis=0)) / x.std(axis=0)
        else:
            x = (x - mean) / std
        return x


def BasicDistance(x, y):
    """
    Calculate the distance between two points

    :param x: Input point coordinate data
    :param y: Input target point coordinate data
    :return: distance matrix
    """
    x = np.float32(x)
    y = np.float32(y)
    dist = distance.cdist(x, y, 'euclidean')
    return dist


def Manhattan_distance(x, y):
    """
    Calculate the Manhattan distance between two points

    :param x: Input point coordinate data
    :param y: Input target point coordinate data
    :return: distance matrix
    """
    return np.float32(np.sum(np.abs(x[:, np.newaxis, :] - y), axis=2))


def init_dataset(data, test_ratio, valid_ratio, x_column, y_column, spatial_column=None, temp_column=None,
                 id_column=None, sample_seed=42, process_fn="minmax_scale", batch_size=32, shuffle=True,
                 use_class=baseDataset,
                 spatial_fun=BasicDistance, temporal_fun=Manhattan_distance, max_val_size=-1, max_test_size=-1,
                 from_for_cv=0, is_need_STNN=False, Reference=None, simple_distance=True, dropna=True):
    """
    Initialize the dataset and return the training set, validation set and test set for the model

    :param data: dataset
    :param test_ratio: test data ratio
    :param valid_ratio: valid data ratio
    :param x_column: input attribute column name
    :param y_column: output attribute column name
    :param spatial_column: spatial attribute column name
    :param temp_column: temporal attribute column name
    :param id_column: id column name
    :param sample_seed: random seed
    :param process_fn: data pre-process function
    :param batch_size: batch size
    :param max_val_size: max valid data size in one injection
    :param max_test_size: max test data size in one injection
    :param shuffle: shuffle data
    :param use_class: dataset class
    :param spatial_fun: spatial distance calculate function
    :param temporal_fun: temporal distance calculate function
    :param from_for_cv: the start index of the data for cross validation
    :param is_need_STNN: whether to use STNN
    :param Reference: reference points to calculate the distance
    :param simple_distance: whether to use simple distance function to calculate the distance
    :return: train dataset, valid dataset, test dataset
    """
    if spatial_fun is None:
        # if dist_fun is None, raise error
        raise ValueError(
            "dist_fun must be a function that can process the data")

    if spatial_column is None:
        # if dist_column is None, raise error
        raise ValueError(
            "dist_column must be a column name in data")
    if dropna:
        oriLen = data.shape[0]
        data.dropna(axis=0,how='any',inplace=True)
        if oriLen > data.shape[0]:
            warnings.warn("Dropping {} {} with missing values. To forbid dropping, you need to set the argument dropna=False".format(oriLen - data.shape[0],'row' if oriLen - data.shape[0] == 1 else 'rows'))
    if id_column is None:
        id_column = ['id']
        if 'id' not in data.columns:
            data['id'] = np.arange(len(data))
        else:
            warnings.warn("id_column is None and use default id column in data", RuntimeWarning)
    np.random.seed(sample_seed)
    data = data.sample(frac=1)  # shuffle data
    scaler_x = None
    scaler_y = None
    # data pre-process
    if process_fn == "minmax_scale":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif process_fn == "standard_scale":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    scaler_params_x = scaler_x.fit(data[x_column])
    scaler_params_y = scaler_y.fit(data[y_column])
    scaler_params = [scaler_params_x, scaler_params_y]
    if process_fn == "minmax_scale":
        print("x_min:" + str(scaler_params_x.data_min_) + ";  x_max:" + str(scaler_params_x.data_max_))
        print("y_min:" + str(scaler_params_y.data_min_) + ";  y_max:" + str(scaler_params_y.data_max_))
    elif process_fn == "standard_scale":
        print("x_mean:" + str(scaler_params_x.mean_) + ";  x_var:" + str(scaler_params_x.var_))
        print("y_mean:" + str(scaler_params_y.mean_) + ";  y_var:" + str(scaler_params_y.var_))

    # data split
    test_data = data[int((1 - test_ratio) * len(data)):]
    train_data = data[:int((1 - test_ratio) * len(data))]
    val_data = train_data[
               int(from_for_cv * valid_ratio * len(train_data)):int((1 + from_for_cv) * valid_ratio * len(train_data))]
    train_data = pandas.concat([train_data[:int(from_for_cv * valid_ratio * len(train_data))],
                                train_data[int((1 + from_for_cv) * valid_ratio * len(train_data)):]])

    # Use the parameters of the dataset to normalize the train_dataset, val_dataset, and test_dataset
    train_dataset = use_class(train_data, x_column, y_column, id_column, is_need_STNN)
    val_dataset = use_class(val_data, x_column, y_column, id_column, is_need_STNN)
    test_dataset = use_class(test_data, x_column, y_column, id_column, is_need_STNN)
    train_dataset.scale(process_fn, scaler_params)
    val_dataset.scale(process_fn, scaler_params)
    test_dataset.scale(process_fn, scaler_params)

    if Reference is None:
        reference_data = train_data
    elif isinstance(Reference, str):
        if Reference == "train":
            reference_data = train_data
        elif Reference == "train_val":
            reference_data = pandas.concat([train_data, val_data])
        else:
            raise ValueError("Reference str must be 'train' or 'train_val'")
    else:
        reference_data = Reference
    if not isinstance(reference_data, pandas.DataFrame):
        raise ValueError("reference_data must be a pandas.DataFrame")
    train_dataset.reference, val_dataset.reference, test_dataset.reference = reference_data, reference_data, reference_data
    train_dataset.spatial_column = val_dataset.spatial_column = test_dataset.spatial_column = spatial_column
    train_dataset.x_column = val_dataset.x_column = test_dataset.x_column = x_column
    train_dataset.y_column = val_dataset.y_column = test_dataset.y_column = y_column
    if not is_need_STNN:
        if simple_distance:
            # if not use STNN, calculate spatial/temporal distance matrix and concatenate them
            train_dataset.distances = spatial_fun(
                train_data[spatial_column].values, reference_data[spatial_column].values)  # 计算train距离矩阵
            val_dataset.distances = spatial_fun(
                val_data[spatial_column].values, reference_data[spatial_column].values)  # 计算val距离矩阵
            test_dataset.distances = spatial_fun(
                test_data[spatial_column].values, reference_data[spatial_column].values)  # 计算test距离矩阵

            if temp_column is not None:
                # if temp_column is not None, calculate temporal distance matrix
                train_dataset.temporal = temporal_fun(
                    train_data[temp_column].values, reference_data[temp_column].values)
                val_dataset.temporal = temporal_fun(
                    val_data[temp_column].values, reference_data[temp_column].values)
                test_dataset.temporal = temporal_fun(
                    test_data[temp_column].values, reference_data[temp_column].values)

                train_dataset.distances = np.concatenate(
                    (train_dataset.distances[:, :, np.newaxis], train_dataset.temporal[:, :, np.newaxis]),
                    axis=2)  # concatenate spatial and temporal distance matrix
                val_dataset.distances = np.concatenate(
                    (val_dataset.distances[:, :, np.newaxis], val_dataset.temporal[:, :, np.newaxis]), axis=2)
                test_dataset.distances = np.concatenate(
                    (test_dataset.distances[:, :, np.newaxis], test_dataset.temporal[:, :, np.newaxis]), axis=2)
        else:
            train_dataset.distances = np.repeat(train_data[spatial_column].values[:, np.newaxis, :],
                                                len(reference_data),
                                                axis=1)
            train_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :],
                                            train_dataset.datasize,
                                            axis=1)
            train_dataset.distances = np.concatenate(
                (train_dataset.distances, np.transpose(train_temp_distance, (1, 0, 2))), axis=2)

            val_dataset.distances = np.repeat(val_data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                              axis=1)
            val_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :], val_dataset.datasize,
                                          axis=1)
            val_dataset.distances = np.concatenate((val_dataset.distances, np.transpose(val_temp_distance, (1, 0, 2))),
                                                   axis=2)

            test_dataset.distances = np.repeat(test_data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                               axis=1)
            test_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :],
                                           test_dataset.datasize,
                                           axis=1)
            test_dataset.distances = np.concatenate(
                (test_dataset.distances, np.transpose(test_temp_distance, (1, 0, 2))), axis=2)
            # if temp_column is not None, calculate temporal point matrix
            if temp_column is not None:
                train_dataset.temporal = np.repeat(train_data[temp_column].values[:, np.newaxis, :],
                                                   len(reference_data),
                                                   axis=1)
                train_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                                train_dataset.datasize,
                                                axis=1)
                train_dataset.temporal = np.concatenate(
                    (train_dataset.temporal, np.transpose(train_temp_temporal, (1, 0, 2))), axis=2)

                val_dataset.temporal = np.repeat(val_data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                                 axis=1)
                val_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                              val_dataset.datasize,
                                              axis=1)
                val_dataset.temporal = np.concatenate(
                    (val_dataset.temporal, np.transpose(val_temp_temporal, (1, 0, 2))),
                    axis=2)

                test_dataset.temporal = np.repeat(test_data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                                  axis=1)
                test_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                               test_dataset.datasize,
                                               axis=1)
                test_dataset.temporal = np.concatenate(
                    (test_dataset.temporal, np.transpose(test_temp_temporal, (1, 0, 2))), axis=2)
            train_dataset.distances = np.concatenate(
                (train_dataset.distances, train_dataset.temporal), axis=2)
            val_dataset.distances = np.concatenate(
                (val_dataset.distances, val_dataset.temporal), axis=2)
            test_dataset.distances = np.concatenate(
                (test_dataset.distances, test_dataset.temporal), axis=2)
    else:
        # if use STNN, calculate spatial/temporal point matrix
        train_dataset.distances = np.repeat(train_data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                            axis=1)
        train_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :], train_dataset.datasize,
                                        axis=1)
        train_dataset.distances = np.concatenate(
            (train_dataset.distances, np.transpose(train_temp_distance, (1, 0, 2))), axis=2)

        val_dataset.distances = np.repeat(val_data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                          axis=1)
        val_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :], val_dataset.datasize,
                                      axis=1)
        val_dataset.distances = np.concatenate((val_dataset.distances, np.transpose(val_temp_distance, (1, 0, 2))),
                                               axis=2)

        test_dataset.distances = np.repeat(test_data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                           axis=1)
        test_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :], test_dataset.datasize,
                                       axis=1)
        test_dataset.distances = np.concatenate(
            (test_dataset.distances, np.transpose(test_temp_distance, (1, 0, 2))), axis=2)
        # if temp_column is not None, calculate temporal point matrix
        if temp_column is not None:
            train_dataset.temporal = np.repeat(train_data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                               axis=1)
            train_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                            train_dataset.datasize,
                                            axis=1)
            train_dataset.temporal = np.concatenate(
                (train_dataset.temporal, np.transpose(train_temp_temporal, (1, 0, 2))), axis=2)

            val_dataset.temporal = np.repeat(val_data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                             axis=1)
            val_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :], val_dataset.datasize,
                                          axis=1)
            val_dataset.temporal = np.concatenate((val_dataset.temporal, np.transpose(val_temp_temporal, (1, 0, 2))),
                                                  axis=2)

            test_dataset.temporal = np.repeat(test_data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                              axis=1)
            test_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :], test_dataset.datasize,
                                           axis=1)
            test_dataset.temporal = np.concatenate(
                (test_dataset.temporal, np.transpose(test_temp_temporal, (1, 0, 2))), axis=2)
    train_dataset.simple_distance = simple_distance
    val_dataset.simple_distance = simple_distance
    test_dataset.simple_distance = simple_distance
    # initialize dataloader for train/val/test dataset
    # set batch_size for train_dataset as batch_size
    # set batch_size for val_dataset as max_val_size
    # set batch_size for test_dataset as max_test_size
    if max_val_size < 0:
        max_val_size = len(val_dataset)
    if max_test_size < 0:
        max_test_size = len(test_dataset)
    if process_fn == "minmax_scale":
        distance_scale = MinMaxScaler()
        temporal_scale = MinMaxScaler()
    else:
        distance_scale = StandardScaler()
        temporal_scale = StandardScaler()
    # scale distance matrix
    train_distance_len = len(train_dataset.distances)
    val_distance_len = len(val_dataset.distances)
    distances = np.concatenate((train_dataset.distances, val_dataset.distances, test_dataset.distances), axis=0)
    distances = distance_scale.fit_transform(distances.reshape(-1, distances.shape[-1])).reshape(distances.shape)
    if process_fn == "minmax_scale":
        distance_scale_param = {"min": distance_scale.data_min_, "max": distance_scale.data_max_}
    else:
        distance_scale_param = {"mean": distance_scale.mean_, "var": distance_scale.var_}
    train_dataset.distances = distances[:train_distance_len]
    val_dataset.distances = distances[train_distance_len:train_distance_len + val_distance_len]
    test_dataset.distances = distances[train_distance_len + val_distance_len:]
    train_dataset.distances_scale_param = val_dataset.distances_scale_param = test_dataset.distances_scale_param = distance_scale_param
    if temp_column is not None:
        temporal = np.concatenate((train_dataset.temporal, val_dataset.temporal, test_dataset.temporal), axis=0)
        temporal = temporal_scale.fit_transform(temporal.reshape(-1, temporal.shape[-1])).reshape(temporal.shape)
        if process_fn == "minmax_scale":
            temporal_scale_param = {"min": temporal_scale.data_min_, "max": temporal_scale.data_max_}
        else:
            temporal_scale_param = {"mean": temporal_scale.mean_, "var": temporal_scale.var_}
        train_dataset.temporal = temporal[:train_distance_len]
        val_dataset.temporal = temporal[train_distance_len:train_distance_len + val_distance_len]
        test_dataset.temporal = temporal[train_distance_len + val_distance_len:]
        train_dataset.temporal_scale_param = val_dataset.temporal_scale_param = test_dataset.temporal_scale_param = temporal_scale_param

    train_dataset.dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataset.dataloader = DataLoader(
        val_dataset, batch_size=max_val_size, shuffle=shuffle)
    test_dataset.dataloader = DataLoader(
        test_dataset, batch_size=max_test_size, shuffle=shuffle)
    train_dataset.batch_size, train_dataset.shuffle = batch_size, shuffle
    val_dataset.batch_size, val_dataset.shuffle = max_val_size, shuffle
    test_dataset.batch_size, test_dataset.shuffle = max_test_size, shuffle
    return train_dataset, val_dataset, test_dataset


def init_dataset_cv(data, test_ratio, k_fold, x_column, y_column, spatial_column=None, temp_column=None,
                    id_column=None,
                    sample_seed=100,
                    process_fn="minmax_scale", batch_size=32, shuffle=True, use_class=baseDataset,
                    spatial_fun=BasicDistance, temporal_fun=Manhattan_distance, max_val_size=-1, max_test_size=-1,
                    is_need_STNN=False, Reference=None, simple_distance=True):
    """
    initialize dataset for cross validation


    :param data: input data
    :param test_ratio: test set ratio
    :param k_fold:  k of k-fold
    :param x_column: attribute column name
    :param y_column: label column name
    :param spatial_column: spatial distance column name
    :param temp_column: temporal distance column name
    :param id_column: id column name
    :param sample_seed: random seed
    :param process_fn: data process function
    :param batch_size: batch size
    :param shuffle: shuffle or not
    :param use_class: dataset class
    :param spatial_fun: spatial distance calculate function
    :param temporal_fun: temporal distance calculate function
    :param max_val_size: validation set size
    :param max_test_size: test set size
    :param is_need_STNN: whether need STNN
    :param Reference: reference data
    :param simple_distance: is simple distance
    :return: cv_data_set, test_dataset
    """
    cv_data_set = []
    valid_ratio = (1 - test_ratio) / k_fold
    test_dataset = None
    for i in range(k_fold):
        train_dataset, val_dataset, test_dataset = init_dataset(data, test_ratio, valid_ratio, x_column, y_column,
                                                                spatial_column,
                                                                temp_column,
                                                                id_column,
                                                                sample_seed,
                                                                process_fn, batch_size, shuffle, use_class,
                                                                spatial_fun, temporal_fun, max_val_size, max_test_size,
                                                                i, is_need_STNN, Reference, simple_distance)
        cv_data_set.append((train_dataset, val_dataset))
    return cv_data_set, test_dataset


# TODO Not finished
# def init_dataset_with_dist_frame(data, train_ratio, valid_ratio, x_column, y_column, id_column, dist_frame=None,
#                                  process_fn="minmax_scale", batch_size=32, shuffle=True, use_class=baseDataset):
#     train_data, val_data, test_data = np.split(data.sample(frac=1),
#                                                [int(train_ratio * len(data)),
#                                                 int((train_ratio + valid_ratio) * len(data))])
#
#     # 初始化train_dataset,val_dataset,test_dataset
#     train_dataset = use_class(train_data, x_column, y_column, process_fn)
#     val_dataset = use_class(val_data, x_column, y_column, process_fn)
#     test_dataset = use_class(test_data, x_column, y_column, process_fn)
#
#     dist_frame.columns = ['id1', 'id2', 'dis']
#     dist_frame = dist_frame.set_index(['id1', 'id2'])[
#         'dis'].unstack().reset_index().drop('id1', axis=1)
#
#     train_ids = train_data[id_column[0]].tolist()
#     val_ids = val_data[id_column[0]].tolist()
#     test_ids = test_data[id_column[0]].tolist()
#
#     train_dataset.distances = np.float32(
#         dist_frame[dist_frame.index.isin(train_ids)][train_ids].values)
#     val_dataset.distances = np.float32(
#         dist_frame[dist_frame.index.isin(val_ids)][train_ids].values)
#     test_dataset.distances = np.float32(
#         dist_frame[dist_frame.index.isin(test_ids)][train_ids].values)
#
#     train_dataset.dataloader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_dataset.dataloader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=shuffle)
#     test_dataset.dataloader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=shuffle)
#
#     return train_dataset, val_dataset, test_dataset


def init_predict_dataset(data, train_dataset, x_column, spatial_column=None, temp_column=None,
                         process_fn="minmax_scale", scale_sync=True, use_class=predictDataset,
                         spatial_fun=BasicDistance, temporal_fun=Manhattan_distance, max_size=-1, is_need_STNN=False):
    """
    initialize predict dataset

    :param data: input data
    :param train_dataset: train data
    :param x_column: attribute column name
    :param spatial_column: spatial distance column name
    :param temp_column: temporal distance column name
    :param process_fn: data process function
    :param scale_sync: scale sync or not
    :param max_size: max size of predict dataset
    :param use_class: dataset class
    :param spatial_fun: spatial distance calculate function
    :param temporal_fun: temporal distance calculate function
    :param is_need_STNN: is need STNN or not
    :return: predict_dataset
    """
    if spatial_fun is None:
        # if dist_fun is None, raise error
        raise ValueError(
            "dist_fun must be a function that can process the data")

    if spatial_column is None:
        # if dist_column is None, raise error
        raise ValueError(
            "dist_column must be a column name in data")

    # initialize the predict_dataset
    if train_dataset.scale_fn == "minmax_scale":
        process_params = [[train_dataset.x_scale_info['min'], train_dataset.x_scale_info['max']]]
    elif train_dataset.scale_fn == "standard_scale":
        process_params = [[train_dataset.x_scale_info['mean'], train_dataset.x_scale_info['std']]]
    else:
        raise ValueError("scale_fn must be minmax_scale or standard_scale")
    # print("ProcessParams:",process_params)
    if scale_sync:
        predict_dataset = use_class(data=data, x_column=x_column, process_fn=process_fn, scale_info=process_params,
                                    is_need_STNN=is_need_STNN)
    else:
        predict_dataset = use_class(data=data, x_column=x_column, process_fn=process_fn, is_need_STNN=is_need_STNN)

    # train_data = train_dataset.dataframe
    reference_data = train_dataset.reference

    if not is_need_STNN:
        # if not use STNN, calculate spatial/temporal distance matrix and concatenate them
        if train_dataset.simple_distance:
            predict_dataset.distances = spatial_fun(
                data[spatial_column].values, reference_data[spatial_column].values)

            if temp_column is not None:
                # if temp_column is not None, calculate temporal distance matrix
                predict_dataset.temporal = temporal_fun(
                    data[temp_column].values, reference_data[temp_column].values)

                predict_dataset.distances = np.concatenate(
                    (predict_dataset.distances[:, :, np.newaxis], predict_dataset.temporal[:, :, np.newaxis]),
                    axis=2)  # concatenate spatial and temporal distance matrix
        else:
            predict_dataset.distances = np.repeat(data[spatial_column].values[:, np.newaxis, :],
                                                  len(reference_data),
                                                  axis=1)
            predict_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :],
                                              predict_dataset.datasize,
                                              axis=1)
            predict_dataset.distances = np.concatenate(
                (predict_dataset.distances, np.transpose(predict_temp_distance, (1, 0, 2))), axis=2)

            if temp_column is not None:
                predict_dataset.temporal = np.repeat(data[temp_column].values[:, np.newaxis, :],
                                                     len(reference_data),
                                                     axis=1)
                predict_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                                  predict_dataset.datasize,
                                                  axis=1)
                predict_dataset.temporal = np.concatenate(
                    (predict_dataset.temporal, np.transpose(predict_temp_temporal, (1, 0, 2))), axis=2)
            predict_dataset.distances = np.concatenate(
                (predict_dataset.distances, predict_dataset.temporal), axis=2)

    else:
        # if use STNN, calculate spatial/temporal point matrix
        # spatial distances matrix
        predict_dataset.distances = np.repeat(data[spatial_column].values[:, np.newaxis, :], len(reference_data),
                                              axis=1)
        predict_temp_distance = np.repeat(reference_data[spatial_column].values[:, np.newaxis, :],
                                          predict_dataset.datasize,
                                          axis=1)
        predict_dataset.distances = np.concatenate(
            (predict_dataset.distances, np.transpose(predict_temp_distance, (1, 0, 2))), axis=2)

        # temporal distances matrix
        if temp_column is not None:
            predict_dataset.temporal = np.repeat(data[temp_column].values[:, np.newaxis, :], len(reference_data),
                                                 axis=1)
            predict_temp_temporal = np.repeat(reference_data[temp_column].values[:, np.newaxis, :],
                                              predict_dataset.datasize,
                                              axis=1)
            predict_dataset.temporal = np.concatenate(
                (predict_dataset.temporal, np.transpose(predict_temp_temporal, (1, 0, 2))), axis=2)
    if process_fn == "minmax_scale":
        predict_dataset.distances = predict_dataset.minmax_scaler(predict_dataset.distances,
                                                                  train_dataset.distances_scale_param['min'],
                                                                  train_dataset.distances_scale_param['max'])
    else:
        predict_dataset.distances = predict_dataset.standard_scaler(predict_dataset.distances,
                                                                    train_dataset.distances_scale_param['mean'],
                                                                    train_dataset.distances_scale_param['var'])
    # initialize dataloader for train/val/test dataset
    if max_size < 0:
        max_size = len(predict_dataset)
    predict_dataset.dataloader = DataLoader(
        predict_dataset, batch_size=max_size, shuffle=False)

    return predict_dataset


def load_dataset(directory, use_class=baseDataset):
    dataset = use_class()
    dataset.read(directory)
    dataset.dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=dataset.shuffle)
    return dataset
