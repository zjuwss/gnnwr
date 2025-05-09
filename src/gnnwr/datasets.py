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
        return torch.tensor(self.distances[index], dtype=torch.float), \
                torch.tensor(self.x_data[index],dtype=torch.float), \
                torch.tensor(self.y_data[index], dtype=torch.float), \
                torch.tensor(self.id_data[index], dtype=torch.float)


    def scale(self, scale_fn, scale_params):
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
        x_scale_params = scale_params[0]
        y_scale_params = scale_params[1]        
        
        if not x_scale_params:
            x_scale_params = None
        if not y_scale_params:
            y_scale_params = None

        self.x_scale_info,self.y_scale_info = x_scale_params, y_scale_params
            
        if scale_fn == "minmax_scale":
            self.scale_fn = "minmax_scale"
            if x_scale_params is not None:
                self.x_data = (self.x_data - x_scale_params["min"]) / (x_scale_params["max"] - x_scale_params["min"])
            if y_scale_params is not None:
                self.y_data = (self.y_data - y_scale_params["min"]) / (y_scale_params["max"] - y_scale_params["min"])
        elif scale_fn == "standard_scale":
            self.scale_fn = "standard_scale"
            if x_scale_params is not None:
                self.x_data = (self.x_data - x_scale_params['mean']) / np.sqrt(x_scale_params["var"])
            if y_scale_params is not None:
                self.y_data = (self.y_data - y_scale_params['mean']) / np.sqrt(y_scale_params["var"])

        self.getScaledDataframe() # Calculate ScaledDataframe

        self.x_data = np.concatenate((self.x_data, np.ones(
            (self.datasize, 1))), axis=1)

    def getScaledDataframe(self):
        """
        get the scaled dataframe and save it in ``scaledDataframe``
        """
        columns = np.concatenate((self.x, self.y), axis=0)
        scaledData = np.concatenate((self.x_data, self.y_data), axis=1)
        self.scaledDataframe = pd.DataFrame(scaledData, columns=columns)

    def rescale(self, x, y):
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
            if x is not None and self.x_scale_info is not None:
                x = np.multiply(x, self.x_scale_info["max"] - self.x_scale_info["min"]) + self.x_scale_info["min"]
            elif self.x_scale_info is None:
                raise ValueError("Invalid x scale info")
            if y is not None and self.y_scale_info is not None:
                y = np.multiply(y, self.y_scale_info["max"] - self.y_scale_info["min"]) + self.y_scale_info["min"]
            elif self.y_scale_info is None:
                raise ValueError("Invalid y scale info")
        elif self.scale_fn == "standard_scale":
            if x is not None and self.x_scale_info is not None:
                x = np.multiply(x, np.sqrt(self.x_scale_info["var"])) + self.x_scale_info["mean"]
            elif self.x_scale_info is None:
                raise ValueError("Invalid x scale info")
            if y is not None and self.y_scale_info is not None:
                y = np.multiply(y, np.sqrt(self.y_scale_info["var"])) + self.y_scale_info["mean"]
            elif self.y_scale_info is None:
                raise ValueError("Invalid y scale info")
        else:
            raise ValueError("invalid process_fn")
        return x, y

    def save(self, dirname, exist_ok=False):
        """
        save the dataset

        :param dirname: save directory
        """
        if os.path.exists(dirname) and not exist_ok:
            raise ValueError("dir is already exists")
        if self.dataframe is None:
            raise ValueError("dataframe is None")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        x_scale_info = {}
        y_scale_info = {}
        if self.x_scale_info is not None:
            for key, value in self.x_scale_info.items():
                x_scale_info[key] = value.tolist()
        if self.y_scale_info is not None:
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
        self.x = dataset_info.get("x", None)
        self.y = dataset_info.get("y", None)
        self.id = dataset_info.get("id", None)
        self.batch_size = dataset_info.get("batch_size", None)
        self.shuffle = dataset_info.get("shuffle", None)
        self.is_need_STNN = dataset_info.get("is_need_STNN", None)
        self.scale_fn = dataset_info.get("scale_fn", None)
        self.simple_distance = dataset_info.get("simple_distance", None)
        self.x_scale_info = json.loads(dataset_info.get("x_scale_info", None))
        self.y_scale_info = json.loads(dataset_info.get("y_scale_info", None))
        self.distances_scale_param = json.loads(dataset_info.get("distance_scale_info", None))
        x_scale_info = self.x_scale_info
        y_scale_info = self.y_scale_info
        if self.x_scale_info is not None:
            for key, value in x_scale_info.items():
                self.x_scale_info[key] = np.array(value)
        if self.y_scale_info is not None:
            for key, value in y_scale_info.items():
                self.y_scale_info[key] = np.array(value)
        # read the distance matrix
        self.distances = np.load(os.path.join(dirname, "distances.npy")).astype(np.float32)
        # read dataframe
        self.dataframe = pd.read_csv(os.path.join(dirname, "dataframe.csv"))
        self.x_data = self.dataframe[self.x].astype(np.float32).values
        self.datasize = self.x_data.shape[0]
        self.y_data = self.dataframe[self.y].astype(np.float32).values
        self.id_data = self.dataframe[self.id].astype(np.int64).values
        self.coefsize = len(self.x) + 1
        self.scale(self.scale_fn, [self.x_scale_info, self.y_scale_info])



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
            self.x_scale_info = scale_info[0]  # scale information of x_data
            self.y_scale_info = scale_info[1]
            self.use_scale_info = True
        else:
            self.x_scale_info, self.y_scale_info = None, None
            self.use_scale_info = False
        
        # 数据预处理
        self.scale_fn = process_fn
        if process_fn == "minmax_scale":
            if self.use_scale_info:
                self.x_data = self.minmax_scaler(self.x_data, self.x_scale_info["min"], self.x_scale_info["max"])
            else:
                self.x_data = self.minmax_scaler(self.x_data)
        elif process_fn == "standard_scale":
            if self.use_scale_info:
                self.x_data = self.standard_scaler(self.x_data, self.x_scale_info["mean"], np.sqrt(self.x_scale_info["var"]))
            else:
                self.x_data = self.standard_scaler(self.x_data)
        else:
            raise ValueError("invalid process_fn")

        self.x_data = np.concatenate((self.x_data, np.ones((self.datasize, 1))), axis=1)

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
                              torch.tensor(self.temporal[index], dtype=torch.float)), dim=-1), \
                    torch.tensor(self.x_data[index], dtype=torch.float)
        return torch.tensor(self.distances[index], dtype=torch.float), \
                torch.tensor(self.x_data[index], dtype=torch.float)

    def rescale(self, x, y):
        """
        rescale the attribute data

        :param x: Input attribute data
        :return: rescaled attribute data
        """
        if self.scale_fn == "minmax_scale":
            if x is not None and self.x_scale_info is not None:
                x = x * (self.x_scale_info["max"] - self.x_scale_info["min"]) + self.x_scale_info["min"]
            elif self.x_scale_info is None:
                raise ValueError("Invalid x scale info")
            if y is not None and  self.y_scale_info is not None:
                y = y * (self.y_scale_info["max"] - self.y_scale_info["min"]) + self.y_scale_info["min"]
            elif self.y_scale_info is None:
                raise ValueError("Invalid y scale info")
    
        elif self.scale_fn == "standard_scale":
            if x is not None and self.x_scale_info is not None:
                x = x * np.sqrt(self.x_scale_info["var"]) + self.x_scale_info["mean"]
            elif self.x_scale_info is None:
                raise ValueError("Invalid x scale info")
            if y is not None and self.y_scale_info is not None:
                y = y * np.sqrt(self.y_scale_info["var"]) + self.y_scale_info["mean"]
            elif self.y_scale_info is None:
                raise ValueError("Invalid y scale info")
        else:
            raise ValueError("invalid process_fn")

        return x,y

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


def ManhattanDistance(x, y):
    """
    Calculate the Manhattan distance between two points

    :param x: Input point coordinate data
    :param y: Input target point coordinate data
    :return: distance matrix
    """
    return np.float32(np.sum(np.abs(x[:, np.newaxis, :] - y), axis=2))


def init_dataset(data, 
                 test_ratio,
                 valid_ratio,
                 x_column,
                 y_column,
                 spatial_column=None,
                 temp_column=None,
                 id_column=None,
                 sample_seed=42,
                 process_fn="minmax_scale",
                 process_var = ["x"],
                 batch_size=32,
                 shuffle=True,
                 use_model="gnnwr",
                 spatial_fun=BasicDistance,
                 temporal_fun=ManhattanDistance,
                 max_val_size=-1,
                 max_test_size=-1,
                 from_for_cv=0,
                 is_need_STNN=False,
                 Reference=None,
                 simple_distance=True,
                 dropna=True
                 ):
    r"""
    Initialize the dataset and return the training set, validation set, and test set for the model.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be initialized.
    test_ratio : float
        The ratio of test data.
    valid_ratio : float
        The ratio of validation data.
    x_column : list
        The name of the input attribute column.
    y_column : list
        The name of the output attribute column.
    spatial_column : list
        The name of the spatial attribute column.
    temp_column : list
        The name of the temporal attribute column.
    id_column : list
        The name of the ID column.
    sample_seed : int
        The random seed for sampling.
    process_fn : callable
        The data pre-processing function.
    batch_size : int
        The size of the batch.
    shuffle : bool
        Whether to shuffle the data.
    use_model : str
        The model to be used, e.g., "gnnwr", "gnnwr spnn", "gtnnwr", "gtnnwr stpnn".
    spatial_fun : callable
        The function for calculating spatial distance.
    temporal_fun : callable
        The function for calculating temporal distance.
    max_val_size : int
        The maximum size of the validation data in one injection.
    max_test_size : int
        The maximum size of the test data in one injection.
    from_for_cv : int
        The start index of the data for cross-validation.
    is_need_STNN : bool
        A flag indicating whether to use STNN.
    Reference : Union[str, pandas.DataFrame]
        Reference points for calculating the distance. It can be a string ["train", "train_val"] or a pandas DataFrame.
    simple_distance : bool
        A flag indicating whether to use a simple distance function for calculation.
    dropna : bool
        A flag indicating whether to drop NaN values.

    Returns
    -------
    train_dataset : baseDataset
        The training dataset.
    valid_dataset : baseDataset
        The validation dataset.
    test_dataset : baseDataset
        The test dataset.
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
        data.dropna(axis=0, how='any', inplace=True)
        if oriLen > data.shape[0]:
            warnings.warn(
                "Dropping {} {} with missing values. To forbid dropping, you need to set the argument dropna=False".format(
                    oriLen - data.shape[0], 'row' if oriLen - data.shape[0] == 1 else 'rows'))
    if id_column is None:
        id_column = ['id']
        if 'id' not in data.columns:
            data['id'] = np.arange(len(data))
        else:
            warnings.warn("id_column is None and use default id column in data", RuntimeWarning)
    np.random.seed(sample_seed)
    data = data.sample(frac=1)  # shuffle data
    # data split
    test_data = data[int((1 - test_ratio) * len(data)):]
    train_data = data[:int((1 - test_ratio) * len(data))]
    val_data = train_data[
               int(from_for_cv * valid_ratio * len(train_data)):int((1 + from_for_cv) * valid_ratio * len(train_data))]
    train_data = pandas.concat([train_data[:int(from_for_cv * valid_ratio * len(train_data))],
                                train_data[int((1 + from_for_cv) * valid_ratio * len(train_data)):]])
    return init_dataset_split(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        x_column=x_column,
        y_column=y_column,
        spatial_column=spatial_column,
        temp_column=temp_column,
        id_column=id_column,
        process_fn=process_fn,
        process_var = process_var,
        batch_size=batch_size,
        shuffle=shuffle,
        use_model=use_model,
        spatial_fun=spatial_fun,
        temporal_fun=temporal_fun,
        max_val_size=max_val_size,
        max_test_size=max_test_size,
        is_need_STNN=is_need_STNN,
        Reference=Reference,
        simple_distance=simple_distance,
        dropna=dropna
    )

    

def init_dataset_split(train_data, 
                    val_data,
                    test_data,
                    x_column,
                    y_column,
                    spatial_column=None,
                    temp_column=None,
                    id_column=None,
                    process_fn="minmax_scale",
                    process_var = ["x"],
                    batch_size=32,
                    shuffle=True,
                    use_model="gnnwr",
                    spatial_fun=BasicDistance,
                    temporal_fun=ManhattanDistance,
                    max_val_size=-1,
                    max_test_size=-1,
                    is_need_STNN=False,
                    Reference=None,
                    simple_distance=True,
                    dropna=True
                    ):
    r"""
    Initialize the dataset and return the training set, validation set, and test set for the model.

    Parameters
    ----------
    train_data : pandas.DataFrame
        The dataset to be initialized.
    valid_data : pandas.DataFrame
        The dataset to be initialized.
    test_data : pandas.DataFrame
        The dataset to be initialized.
    x_column : list
        The name of the input attribute column.
    y_column : list
        The name of the output attribute column.
    spatial_column : list
        The name of the spatial attribute column.
    temp_column : list
        The name of the temporal attribute column.
    id_column : list
        The name of the ID column.
    sample_seed : int
        The random seed for sampling.
    process_fn : callable
        The data pre-processing function.
    batch_size : int
        The size of the batch.
    shuffle : bool
        Whether to shuffle the data.
    use_model : str
        The model to be used, e.g., "gnnwr", "gnnwr spnn", "gtnnwr", "gtnnwr stpnn".
    spatial_fun : callable
        The function for calculating spatial distance.
    temporal_fun : callable
        The function for calculating temporal distance.
    max_val_size : int
        The maximum size of the validation data in one injection.
    max_test_size : int
        The maximum size of the test data in one injection.
    from_for_cv : int
        The start index of the data for cross-validation.
    is_need_STNN : bool
        A flag indicating whether to use STNN.
    Reference : Union[str, pandas.DataFrame]
        Reference points for calculating the distance. It can be a string ["train", "train_val"] or a pandas DataFrame.
    simple_distance : bool
        A flag indicating whether to use a simple distance function for calculation.
    dropna : bool
        A flag indicating whether to drop NaN values.

    Returns
    -------
    train_dataset : baseDataset
        The training dataset.
    valid_dataset : baseDataset
        The validation dataset.
    test_dataset : baseDataset
        The test dataset.
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
        # train_data
        oriLen_train = train_data.shape[0]
        train_data = train_data.dropna(axis=0)

        if oriLen_train > train_data.shape[0]:
            warnings.warn(
                "Dropping {} {} with missing values. To forbid dropping, you need to set the argument dropna=False".format(
                    oriLen_train - train_data.shape[0], 'row' if oriLen_train - train_data.shape[0] == 1 else 'rows'))
        # val_data
        oriLen_val = val_data.shape[0]
        val_data = val_data.dropna(axis=0)
        if oriLen_val > val_data.shape[0]:
            warnings.warn(
                "Dropping {} {} with missing values. To forbid dropping, you need to set the argument dropna=False".format(
                    oriLen_val - val_data.shape[0], 'row' if oriLen_val - val_data.shape[0] == 1 else 'rows'))  
        # test_data
        oriLen_test = test_data.shape[0]
        test_data = test_data.dropna(axis=0)
        if oriLen_test > test_data.shape[0]:
            warnings.warn(
                "Dropping {} {} with missing values. To forbid dropping, you need to set the argument dropna=False".format(
                    oriLen_test - test_data.shape[0], 'row' if oriLen_test - test_data.shape[0] == 1 else 'rows'))  
    
    data = pd.concat((train_data, val_data, test_data))

    data["__belong__"] = np.concatenate((
        np.full(len(train_data),"train"),
        np.full(len(val_data),"val"),
        np.full(len(test_data),"test"),
    )
    )

    if id_column is None:
        id_column = ['id']
        if 'id' not in data.columns:
            data['id'] = np.arange(len(data))
        else:
            warnings.warn("id_column is None and use default id column in data", RuntimeWarning)
    
    train_data = data[:len(train_data)].copy()
    val_data = data[len(train_data):len(train_data)+len(val_data)].copy()
    test_data = data[len(train_data)+len(val_data):].copy()
    

    scaler_x = None
    scaler_y = None
    # data pre-process
    if process_fn == "minmax_scale":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif process_fn == "standard_scale":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()


    scaler_params_x = scaler_x.fit(train_data[x_column])
    scaler_params_y = scaler_y.fit(train_data[y_column])
    
    scaler_params = [None, None]
    # convert Scaler to Scale Params
    if process_fn == "minmax_scale":
        def cvtparams(Scaler):
            return {"max":Scaler.data_max_,"min":Scaler.data_min_}
    elif process_fn == "standard_scale":
        def cvtparams(Scaler):
            return {"mean":Scaler.mean_,"var":Scaler.var_}
    
    if "x" in process_var:
        scaler_params[0] = cvtparams(scaler_params_x)
    if "y" in process_var:
        scaler_params[1] = cvtparams(scaler_params_y)
    
    # Use the parameters of the dataset to normalize the train_dataset, val_dataset, and test_dataset
    if use_model in ["gnnwr", "gnnwr spnn", "gtnnwr", "gtnnwr stpnn"]:
        train_dataset = baseDataset(train_data, x_column, y_column, id_column, is_need_STNN)
        val_dataset = baseDataset(val_data, x_column, y_column, id_column, is_need_STNN)
        test_dataset = baseDataset(test_data, x_column, y_column, id_column, is_need_STNN)
    else:
        # Other dataset will be added soon
        raise ValueError("invalid use_model")
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
    if use_model == "gnnwr":
        train_dataset.distances, val_dataset.distances, test_dataset.distances = _init_gnnwr_distance(
            reference_data[spatial_column].values, train_data[spatial_column].values, val_data[spatial_column].values,
            test_data[spatial_column].values, spatial_fun
        )
    elif use_model == "gtnnwr":
        assert temp_column is not None, "temp_column must be not None in gtnnwr"
        train_dataset.distances, val_dataset.distances, test_dataset.distances = _init_gtnnwr_distance(
            [reference_data[spatial_column].values,reference_data[temp_column].values],
            [train_data[spatial_column].values, train_data[temp_column].values],
            [val_data[spatial_column].values, val_data[temp_column].values],
            [test_data[spatial_column].values, test_data[temp_column].values],
            spatial_fun, temporal_fun
        )
    elif use_model == "gnnwr spnn":
        train_dataset.distances, val_dataset.distances, test_dataset.distances = _init_gnnwr_spnn_distance(
            reference_data[spatial_column].values, train_data[spatial_column].values, val_data[spatial_column].values,
            test_data[spatial_column].values
        )
        train_dataset.is_need_STNN, val_dataset.is_need_STNN, test_dataset.is_need_STNN = True, True, True
    elif use_model == "gtnnwr stpnn":
        assert temp_column is not None, "temp_column must be not None in gtnnwr"
        train_points, val_points, test_points = _init_gtnnwr_stpnn_distance(
            reference_data[spatial_column + temp_column].values, train_data[spatial_column + temp_column].values,
            val_data[spatial_column + temp_column].values, test_data[spatial_column + temp_column].values
        )
        train_dataset.distances, train_dataset.temporal = train_points
        val_dataset.distances, val_dataset.temporal = val_points
        test_dataset.distances, test_dataset.temporal = test_points
        train_dataset.is_need_STNN, val_dataset.is_need_STNN, test_dataset.is_need_STNN = True, True, True
    # Other calculation methods can be added here.

    train_dataset.simple_distance = simple_distance
    val_dataset.simple_distance = simple_distance
    test_dataset.simple_distance = simple_distance


    if process_fn == "minmax_scale":
        distance_scale = MinMaxScaler()
        temporal_scale = MinMaxScaler()
    else:
        distance_scale = StandardScaler()
        temporal_scale = StandardScaler()
    # scale distance matrix
    distances = train_dataset.distances
    distances = distance_scale.fit_transform(distances.reshape(-1, distances.shape[-1])).reshape(distances.shape)

    train_dataset.distances = distance_scale.transform(train_dataset.distances.reshape(-1, train_dataset.distances.shape[-1])).reshape(train_dataset.distances.shape)
    val_dataset.distances = distance_scale.transform(val_dataset.distances.reshape(-1, val_dataset.distances.shape[-1])).reshape(val_dataset.distances.shape)
    test_dataset.distances = distance_scale.transform(test_dataset.distances.reshape(-1, test_dataset.distances.shape[-1])).reshape(test_dataset.distances.shape)
    
    if process_fn == "minmax_scale":
        distance_scale_param = {"min": distance_scale.data_min_, "max": distance_scale.data_max_}
    else:
        distance_scale_param = {"mean": distance_scale.mean_, "var": distance_scale.var_}    
    train_dataset.distances_scale_param = val_dataset.distances_scale_param = test_dataset.distances_scale_param = distance_scale_param
    
    if train_dataset.temporal is not None and val_dataset.temporal is not None and test_dataset.temporal is not None:
        temporal = train_dataset.temporal
        temporal = temporal_scale.fit_transform(temporal.reshape(-1, temporal.shape[-1])).reshape(temporal.shape)

        train_dataset.temporal = temporal_scale.transform(train_dataset.temporal.reshape(-1, train_dataset.temporal.shape[-1])).reshape(train_dataset.temporal.shape)
        val_dataset.temporal = temporal_scale.transform(val_dataset.temporal.reshape(-1, val_dataset.temporal.shape[-1])).reshape(val_dataset.temporal.shape)
        test_dataset.temporal = temporal_scale.transform(test_dataset.temporal.reshape(-1, test_dataset.temporal.shape[-1])).reshape(test_dataset.temporal.shape)

        if process_fn == "minmax_scale":
            temporal_scale_param = {"min": temporal_scale.data_min_, "max": temporal_scale.data_max_}
        else:
            temporal_scale_param = {"mean": temporal_scale.mean_, "var": temporal_scale.var_}
        train_dataset.temporal_scale_param = val_dataset.temporal_scale_param = test_dataset.temporal_scale_param = temporal_scale_param
    # initialize dataloader for train/val/test dataset
    # set batch_size for train_dataset as batch_size
    # set batch_size for val_dataset as max_val_size
    # set batch_size for test_dataset as max_test_size
    train_dataset.dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    if max_val_size < 0:
        max_val_size = len(val_dataset)
    if max_test_size < 0:
        max_test_size = len(test_dataset)
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
                    spatial_fun=BasicDistance, temporal_fun=ManhattanDistance, max_val_size=-1, max_test_size=-1,
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


def init_predict_dataset(data, 
                         train_dataset, 
                         x_column, 
                         spatial_column=None, 
                         temp_column=None,
                         process_fn="minmax_scale", 
                         scale_sync=True, 
                         use_model="gnnwr",
                         use_class=predictDataset,
                         spatial_fun=BasicDistance, 
                         temporal_fun=ManhattanDistance, 
                         batch_size=-1, 
                         is_need_STNN=False):
    """
    initialize predict dataset

    :param data: input data
    :param train_dataset: train data
    :param x_column: attribute column name
    :param spatial_column: spatial distance column name
    :param temp_column: temporal distance column name
    :param process_fn: data process function
    :param scale_sync: scale sync or not
    :param batch_size: max size of predict dataset
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
    if scale_sync:
        predict_dataset = use_class(data=data, 
                                    x_column=x_column, 
                                    process_fn=process_fn, 
                                    scale_info=[train_dataset.x_scale_info, train_dataset.y_scale_info],
                                    is_need_STNN=is_need_STNN)
    else:
        predict_dataset = use_class(data=data, x_column=x_column, process_fn=process_fn, is_need_STNN=is_need_STNN)

    # calculate distance
    predict_data = predict_dataset.dataframe
    reference_data = train_dataset.reference

    if use_model == "gnnwr":
        predict_dataset.distances = _init_gnnwr_distance_pred(
            reference_data[spatial_column].values, predict_data[spatial_column].values,spatial_fun
        )
    elif use_model == "gtnnwr":
        assert temp_column is not None, "temp_column must be not None in gtnnwr"
        predict_dataset.distances = _init_gtnnwr_distance_pred(
            [reference_data[spatial_column].values,reference_data[temp_column].values],
            [predict_data[spatial_column].values, predict_data[temp_column].values],
            spatial_fun, temporal_fun
        )
    elif use_model == "gnnwr spnn":
        predict_dataset.distances = _init_gnnwr_spnn_distance_pred(
            reference_data[spatial_column].values, predict_data[spatial_column].values
        )
        predict_dataset.is_need_STNN = True
    elif use_model == "gtnnwr stpnn":
        assert temp_column is not None, "temp_column must be not None in gtnnwr"
        pred_points = _init_gtnnwr_stpnn_distance_pred(
            reference_data[spatial_column + temp_column].values, predict_data[spatial_column + temp_column].values,
        )
        predict_dataset.distances, predict_dataset.temporal = pred_points
        predict_dataset.is_need_STNN = True

    # distance preprocess
    distance_shape = predict_dataset.distances.shape

    if process_fn == "minmax_scale":
        predict_dataset.distances = predict_dataset.minmax_scaler(predict_dataset.distances.reshape(-1,distance_shape[-1]),
                                                                  train_dataset.distances_scale_param['min'],
                                                                  train_dataset.distances_scale_param['max']).reshape(distance_shape)
    else:
        predict_dataset.distances = predict_dataset.standard_scaler(predict_dataset.distances.reshape(-1,distance_shape[-1]),
                                                                    train_dataset.distances_scale_param['mean'],
                                                                    train_dataset.distances_scale_param['var']).reshape(distance_shape)
    # initialize dataloader for train/val/test dataset
    if batch_size < 0:
        batch_size = len(predict_dataset)
    predict_dataset.dataloader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    return predict_dataset


def load_dataset(directory, use_class=baseDataset):
    dataset = use_class()
    dataset.read(directory)
    dataset.dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=dataset.shuffle)
    return dataset

# To make the distance calculation clearer, each method is separated into independent functions here.
# TODO: fix comment
def _init_gnnwr_distance(refer_data, train_data, val_data, test_data, spatial_fun=BasicDistance):
    r"""
    Parameters
    ----------
    refer_data : numpy.nDarray
        Reference points for calculating the distance.
    train_data : object
        The data subset used for training the model.
    val_data : object
        The data subset used for validating the model during training.
    test_data : object
        The data subset used for testing the model after training.

    Returns
    -------
    train_distance : numpy.nDarray
        matrix of distance between points in train_data and Reference points
    val_distance : numpy.nDarray
        matrix of distance between points in val_data and Reference points
    test_distance : numpy.nDarray
        matrix of distance between points in test_data and Reference points
    """

    train_distance = spatial_fun(train_data, refer_data)
    val_distance = spatial_fun(val_data, refer_data)
    test_distance = spatial_fun(test_data, refer_data)
    return train_distance, val_distance, test_distance

def _init_gnnwr_distance_pred(refer_data, pred_data, spatial_fun=BasicDistance):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    pred_distance = spatial_fun(pred_data, refer_data)
    return pred_distance

def _init_gtnnwr_distance(refer_data,
                          train_data,
                          val_data,
                          test_data,
                          spatial_fun=BasicDistance,
                          temporal_fun=ManhattanDistance):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    refer_s_distance, refer_t_distance = refer_data[0], refer_data[1]
    train_s_distance, train_t_distance = train_data[0], train_data[1]
    val_s_distance, val_t_distance = val_data[0], val_data[1]
    test_s_distance, test_t_distance = test_data[0], test_data[1]

    train_s_distance = spatial_fun(train_s_distance, refer_s_distance)
    val_s_distance = spatial_fun(val_s_distance, refer_s_distance)
    test_s_distance = spatial_fun(test_s_distance, refer_s_distance)
    train_t_distance = temporal_fun(train_t_distance, refer_t_distance)
    val_t_distance = temporal_fun(val_t_distance, refer_t_distance)
    test_t_distance = temporal_fun(test_t_distance, refer_t_distance)
    train_distance = np.concatenate((train_s_distance[:, :, np.newaxis], train_t_distance[:, :, np.newaxis]), axis=2)
    val_distance = np.concatenate((val_s_distance[:, :, np.newaxis], val_t_distance[:, :, np.newaxis]), axis=2)
    test_distance = np.concatenate((test_s_distance[:, :, np.newaxis], test_t_distance[:, :, np.newaxis]), axis=2)
    return train_distance, val_distance, test_distance

def _init_gtnnwr_distance_pred(refer_data,
                            pred_data,
                            spatial_fun=BasicDistance,
                            temporal_fun=ManhattanDistance):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    refer_s_distance, refer_t_distance = refer_data[0], refer_data[1]
    pred_s_distance, pred_t_distance = pred_data[0], pred_data[1]

    pred_s_distance = spatial_fun(pred_s_distance, refer_s_distance)
    pred_t_distance = temporal_fun(pred_t_distance, refer_t_distance)

    pred_distance = np.concatenate((pred_s_distance[:, :, np.newaxis], pred_t_distance[:, :, np.newaxis]), axis=2)
    return pred_distance

def _init_gnnwr_spnn_distance(
        refer_data,
        train_data,
        val_data,
        test_data):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    # calculate point matrix
    matrix_length = len(refer_data)
    train_length,val_length,test_length = len(train_data),len(val_data),len(test_data)
    train_point_matrix = np.repeat(train_data[:, np.newaxis, :], matrix_length, axis=1)
    refer_point_train_matrix = np.repeat(refer_data[:,np.newaxis,:], train_length, axis=1)
    train_point_matrix = np.concatenate(
            (train_point_matrix, np.transpose(refer_point_train_matrix, (1, 0, 2))), axis=2)
    val_point_matrix = np.repeat(val_data[:, np.newaxis, :], matrix_length, axis=1)
    refer_point_val_matrix = np.repeat(refer_data[:,np.newaxis,:], val_length, axis=1)
    val_point_matrix = np.concatenate(
            (val_point_matrix, np.transpose(refer_point_val_matrix, (1, 0, 2))), axis=2)
    test_point_matrix = np.repeat(test_data[:, np.newaxis, :], matrix_length, axis=1)
    refer_point_test_matrix = np.repeat(refer_data[:,np.newaxis,:], test_length, axis=1)
    test_point_matrix = np.concatenate(
            (test_point_matrix, np.transpose(refer_point_test_matrix, (1, 0, 2))), axis=2)
    return train_point_matrix, val_point_matrix, test_point_matrix

def _init_gnnwr_spnn_distance_pred(
        refer_data,
        pred_data):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    # calculate point matrix
    matrix_length = len(refer_data)
    pred_length = len(pred_data)
    pred_point_matrix = np.repeat(pred_data[:, np.newaxis, :], matrix_length, axis=1)
    refer_point_pred_matrix = np.repeat(refer_data[:,np.newaxis,:], pred_length, axis=1)
    pred_point_matrix = np.concatenate(
            (pred_point_matrix, np.transpose(refer_point_pred_matrix, (1, 0, 2))), axis=2)
    return pred_point_matrix

def _init_gtnnwr_stpnn_distance(
        refer_data,
        train_data,
        val_data,
        test_data):
    """
    :param refer_data:
    :param train_data:
    :param val_data:
    :param test_data:
    :return:
    """
    refer_s_point, refer_t_point = refer_data[0], refer_data[1]
    train_s_point, train_t_point = train_data[0], train_data[1]
    val_s_point, val_t_point = val_data[0], val_data[1]
    test_s_point, test_t_point = test_data[0], test_data[1]

    # calculate point matrix
    matrix_length = len(refer_s_point)
    train_length,val_length,test_length = len(train_s_point),len(val_s_point),len(test_s_point)
    train_s_point_matrix = np.repeat(train_s_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_s_point_train_matrix = np.repeat(refer_s_point[:,np.newaxis,:], train_length, axis=1)
    train_s_point_matrix = np.concatenate(
            (train_s_point_matrix, np.transpose(refer_s_point_train_matrix, (1, 0, 2))), axis=2)
    val_s_point_matrix = np.repeat(val_s_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_s_point_val_matrix = np.repeat(refer_s_point[:,np.newaxis,:], val_length, axis=1)
    val_s_point_matrix = np.concatenate(
            (val_s_point_matrix, np.transpose(refer_s_point_val_matrix, (1, 0, 2))), axis=2)
    test_s_point_matrix = np.repeat(test_s_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_s_point_test_matrix = np.repeat(refer_s_point[:,np.newaxis,:], test_length, axis=1)
    test_s_point_matrix = np.concatenate(
            (test_s_point_matrix, np.transpose(refer_s_point_test_matrix, (1, 0, 2))), axis=2)

    train_t_point_matrix = np.repeat(train_t_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_t_point_train_matrix = np.repeat(refer_t_point[:,np.newaxis,:], train_length, axis=1)
    train_t_point_matrix = np.concatenate(
            (train_t_point_matrix, np.transpose(refer_t_point_train_matrix, (1, 0, 2))), axis=2)
    val_t_point_matrix = np.repeat(val_t_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_t_point_val_matrix = np.repeat(refer_t_point[:,np.newaxis,:], val_length, axis=1)
    val_t_point_matrix = np.concatenate(
            (val_t_point_matrix, np.transpose(refer_t_point_val_matrix, (1, 0, 2))), axis=2)
    test_t_point_matrix = np.repeat(test_t_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_t_point_test_matrix = np.repeat(refer_t_point[:,np.newaxis,:], test_length, axis=1)
    test_t_point_matrix = np.concatenate(
            (test_t_point_matrix, np.transpose(refer_t_point_test_matrix, (1, 0, 2))), axis=2)
    return (train_s_point_matrix, train_t_point_matrix), (val_s_point_matrix, val_t_point_matrix), (test_s_point_matrix, test_t_point_matrix)

def _init_gtnnwr_stpnn_distance_pred(
        refer_data,
        pred_data):
    """
    :param refer_data:
    :param pred_data:
    :return:
    """
    refer_s_point, refer_t_point = refer_data[0], refer_data[1]
    pred_s_point, pred_t_point = pred_data[0], pred_data[1]

    # calculate point matrix
    matrix_length = len(refer_s_point)
    pred_length = len(pred_s_point)
    pred_s_point_matrix = np.repeat(pred_s_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_s_point_pred_matrix = np.repeat(refer_s_point[:,np.newaxis,:], pred_length, axis=1)
    pred_s_point_matrix = np.concatenate(
            (pred_s_point_matrix, np.transpose(refer_s_point_pred_matrix, (1, 0, 2))), axis=2)


    pred_t_point_matrix = np.repeat(pred_t_point[:, np.newaxis, :], matrix_length, axis=1)
    refer_t_point_pred_matrix = np.repeat(refer_t_point[:,np.newaxis,:], pred_length, axis=1)
    pred_t_point_matrix = np.concatenate(
            (pred_t_point_matrix, np.transpose(refer_t_point_pred_matrix, (1, 0, 2))), axis=2)
    return (pred_s_point_matrix, pred_t_point_matrix)