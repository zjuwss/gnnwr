r"""
Dataset Module
====================

This module provides dataset classes and utility functions for the `gnnwr` package.
It includes comprehensive data handling capabilities for spatial and spatio-temporal analysis.

Main Components
---------------
**Dataset Classes:**
    * :class:`BaseDataset` - Base dataset class for GNNWR models
    * :class:`GTNNWRDataset` - Dataset class for GTNNWR models
    * :class:`PredictDataset` - Prediction dataset class for model inference

**Initialization Functions:**
    * :func:`create_predict_dataset` - Create prediction dataset for models.
    * :func:`init_dataset` - Initialize datasets for training, validation, and testing by split ratio.
    * :func:`init_dataset_split` - Initialize datasets for train/validation/test by splited dataset.
    * :func:`init_dataset_cv` - Initialize datasets for cross-validation by split ratio.
    * :func:`init_predict_dataset` - Initialize prediction datasets for models.

**Utility Functions:**
    * :func:`_init_gnnwr_dataset` - Initialize GNNWR training/validation/testing datasets
    * :func:`_init_gnnwr_predict_dataset` - Initialize prediction datasets for GNNWR models.
    * :func:`_init_gnnwr_distance` - Initialize spatial distance matrices for GNNWR
    * :func:`_init_gtnnwr_dataset` - Initialize GTNNWR training/validation/testing datasets
    * :func:`_init_gtnnwr_predict_dataset` - Initialize prediction datasets for GTNNWR models.
    * :func:`_init_gtnnwr_distance` - Initialize spatio-temporal distance matrices for GTNNWR

Supported Models
----------------
Currently, this module supports training datasets for the following models:

* **GNNWR** (Geographically Neural Network Weighted Regression)
* **GTNNWR** (Geographically and Temporally Neural Network Weighted Regression)

Will Come Soon:

* **GNNWR-SPNN** (GNNWR with Spatial Proximity Neural Network)
* **GTNNWR-STPNN** (GTNNWR with Spatio-Temporal Proximity Neural Network)

Usage Examples
--------------
**Basic Dataset Creation:**
    >>> import pandas as pd
    >>> from gnnwr.datasets import BaseDataset
    >>> 
    >>> # Load your data
    >>> data = pd.read_csv('your_data.csv')
    >>> 
    >>> # Create GNNWR dataset
    >>> dataset = init_dataset(
    ...     data=data,
    ...     test_ratio=0.2,
    ...     val_ratio=0.1,
    ...     x_columns=['feature1', 'feature2'],
    ...     y_column=['target'],
    ...     spatial_column=['lon', 'lat'],
    ...     id_column=['id']
    ...     model_type='gnnwr'
    ... )

**Saving and Loading Datasets:**
    >>> # Save dataset
    >>> dataset.save('path/to/dataset')
    >>> 
    >>> # Load dataset
    >>> loaded_dataset = BaseDataset.load_dataset('path/to/dataset')

Notes
-----
This module is designed specifically for spatial regression analysis and provides
comprehensive tools for handling geographical and temporal data in neural network
weighted regression models. The implementation follows PyTorch conventions.

For more detailed information about specific classes and functions, please refer
to their individual docstrings and the accompanying documentation.

References
----------
.. [1] `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`__
.. [2] `Geographically and temporally neural network weighted regression for modeling spatiotemporal non-stationary relationships <https://doi.org/10.1080/13658816.2020.1775836>`__


See Also
--------
* :mod:`gnnwr.models` - Model implementations using these datasets
* :mod:`gnnwr.networks` - Neural network architectures
* :mod:`gnnwr.utils` - Utility functions for distance calculations
"""

import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from .utils import BasicDistance, ManhattanDistance

class BaseDataset(Dataset):
    r"""
    ``BaseDataset`` is the base dataset class, 
    representing the core structure for storing data and associated information.
    It provides functionality for data scaling, saving, and loading. 
    Other models' dataset classes need to inherit this class,
    and override the ``transform_network_input``, ``inverse_transform_network_input``, ``__getitem__``, etc. to add specific network input.

    Parameters
    ----------
    data: pandas.DataFrame
        dataframe
    x_columns: list
        independent variable column name
    y_column: list
        dependent variable column name
    id_column: str
        id column name
    **kwargs: dict
        additional arguments, including:
            * bias: bool, optional
                whether to include a bias term, default is True
    """
    def __init__(self,
                 data: pd.DataFrame = None,
                 x_columns: list = None,
                 y_column: list = None,
                 id_column: list = None,
                 **kwargs):
        self.dataframe = data
        self.x_columns = x_columns
        self.y_column = y_column
        self.id_column = id_column
        if data is None:
            self.x_data = None # x_data is independent variables data
            self.y_data = None # y_data is dependent variables data
            self.id_data = None # id_data is id data
        else:
            self.x_data = data[x_columns].astype(np.float32).values if x_columns is not None else None
            self.y_data = data[y_column].astype(np.float32).values if y_column is not None else None
            self.id_data = data[id_column].astype(np.int64).values if id_column is not None else None
            if self.id_data is None:
                raise ValueError("id_column is None, it is required.")
        
        self.network_input = None  # network input, spatial (and temporal) distance matrix

        self.scalers = {
            "x": None,
            "y": None,
            "network_input": None,
        } # scaler of data

        self.scaled_dataframe = None
        self.reference_data = None

        self.kwargs = kwargs
        if self.kwargs.get('bias') is None:
            self.kwargs['bias'] = True
    
    @property
    def datasize(self):
        r"""
        Returns the number of samples in the dataset

        Returns
        -------
        int
            the number of samples
        """
        return self.x_data.shape[0] if self.x_data is not None else -1

    @property
    def coefsize(self):
        r"""
        Returns the number of coefficients including the bias term

        Returns
        -------
        int
            the number of coefficients including the bias term
        """
        if self.kwargs['bias']:
            return len(self.x_columns) + 1 if self.x_columns is not None else -1
        else:
            return len(self.x_columns) if self.x_columns is not None else -1

    def __len__(self):
        r"""
        Returns the number of samples in the dataset

        Returns
        -------
        int
            the number of samples
        """
        return self.datasize

    def __getitem__(self, index):
        r"""
        Retrieve a sample from the dataset by index, 
        returning all data components needed for model training.

        Parameters
        ----------
        index: int
            the index of sample

        Returns
        -------
        network_input: torch.Tensor
            spatial (and temporal) distance matrix
        x_tensor: torch.Tensor
            input features
        y_tensor: torch.Tensor
            target variables
        id_tensor: torch.Tensor
            sample identifiers

        Notes
        -----
        * When bias is enabled, a constant term with value 1 is appended to the input features.
        * network_input need to be set for specific models, 
        e.g., GNNWR, which requires spatial distance matrices.
        """

        if self.kwargs['bias']:
            x_tensor = torch.cat([
                torch.tensor(self.x_data[index], dtype=torch.float),
                torch.ones(1, dtype=torch.float)
            ], dim=0)
        else:
            x_tensor = torch.tensor(self.x_data[index], dtype=torch.float)
        y_tensor = torch.tensor(self.y_data[index], dtype=torch.float)
        id_tensor = torch.tensor(self.id_data[index], dtype=torch.float)
        network_input = torch.tensor(self.network_input[index], dtype=torch.float)

        return network_input, x_tensor, y_tensor, id_tensor

    def get_scaled_dataframe(self):
        r"""
        Get the scaled data and save it in ``scaled_dataframe``
        """
        if self.y_column is None:
            columns = np.concatenate((self.x_columns, self.id_column), axis=0)
            scaled_data = np.concatenate((self.x_data, self.id_data), axis=1)
            self.scaled_dataframe = pd.DataFrame(scaled_data, columns=columns)
        else:
            columns = np.concatenate((self.x_columns, self.y_column, self.id_column), axis=0)
            scaled_data = np.concatenate((self.x_data, self.y_data, self.id_data), axis=1)
            self.scaled_dataframe = pd.DataFrame(scaled_data, columns=columns)

    def transform_x_y(self,
                      x_scaler: object = None,
                      y_scaler: object = None):
        r"""
        Scale the data using the specified scaler
        
        Parameters
        ----------
        x_scaler: object
            scaler of x_data, e.g., MinMaxScaler, StandardScaler, etc.
        y_scaler: object
            scaler of y_data, e.g., MinMaxScaler, StandardScaler, etc.
        """
        if x_scaler is not None:
            self.x_data = x_scaler.transform(self.x_data)
            self.scalers["x"] = x_scaler
        if y_scaler is not None:
            self.y_data = y_scaler.transform(self.y_data)
            self.scalers["y"] = y_scaler
        self.get_scaled_dataframe() # update scaled_dataframe

    def transform_network_input(self,
                                network_input_scaler: object = None):
        r"""
        Scale the network input using the specified scaler,
        this function need to be implemented in the child class for specific network input type, such as ``GTNNWRDataset``.

        
        Parameters
        ----------
        network_input_scaler: object
            scaler of network_input, e.g., MinMaxScaler, StandardScaler, etc.
            it also support dict type, e.g., {"spatial": MinMaxScaler(), "temporal": StandardScaler()}
        """
        if network_input_scaler is not None:
            self.network_input = network_input_scaler.transform(self.network_input.reshape(-1, 1)).reshape(self.network_input.shape)
            self.scalers["network_input"] = network_input_scaler

    def inverse_transform_x_y(self,
                              x: np.ndarray = None, 
                              y: np.ndarray = None):
        """
        inverse transform the data with the scale function and scale parameters

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
        if x is not None:
            if self.scalers.get("x",None) is not None:
                x = self.scalers["x"].inverse_transform(x)
            else:
                warnings.warn(
                    "No x_scaler available. Returning x data unchanged.", 
                    UserWarning,
                )
        if y is not None:
            if self.scalers.get("y",None) is not None:
                y = self.scalers["y"].inverse_transform(y)
            else:
                warnings.warn(
                    "No y_scaler available. Returning y data unchanged.", 
                    UserWarning,
                )

        return x, y
    
    def inverse_transform_network_input(self,
                                        network_input: np.ndarray = None):
        """
        inverse transform the network input with the scale function and scale parameters

        Parameters
        ----------
        network_input: numpy.ndarray
            network input data

        Returns
        -------
        network_input: numpy.ndarray
            rescaled network input data
        """
        if network_input is not None:
            if self.scalers.get("network_input",None) is not None:
                network_input = self.scalers["network_input"].inverse_transform(network_input.reshape(-1, 1)).reshape(network_input.shape)
            else:
                warnings.warn(
                    "No network_input_scaler available. Returning network_input data unchanged.", 
                    UserWarning,
                )
        return network_input

    def save(self, dirname:str, exist_ok:bool=False):
        """
        Save the dataset to the specified directory using joblib for efficient serialization.
        
        This method saves:
        - Dataset metadata in a JSON file
        - Scale information using joblib
        - network_input matrix as .npy files
        - DataFrames as .csv files
        
        Parameters
        ----------
        dirname : str
            Directory path to save the dataset
        exist_ok : bool
            If True, overwrite existing directory (default: False)
        
        Raises
        ------
        ValueError
            If directory exists and exist_ok=False, or if dataframe is None

        Example
        -------
        >>> dataset.save("../demo_result/dataset", exist_ok=True)
        
        """
        if os.path.exists(dirname) and not exist_ok:
            raise ValueError("dir is already exists")

        if self.dataframe is None:
            raise ValueError("dataframe is None")

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        scale_info_path = os.path.join(dirname, "scale_info.joblib")
        joblib.dump(self.scalers, scale_info_path)

        metadata = {
            "x_columns": self.x_columns,
            "y_column": self.y_column,
            "id_column": self.id_column,
        }
        with open(os.path.join(dirname, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # save the network_input matrix
        np.save(os.path.join(dirname, "network_input.npy"),
                self.network_input.astype(np.float32))

        # save dataframe
        self.dataframe.to_csv(os.path.join(dirname, "dataframe.csv"), index=False)
        
        self.scaled_dataframe.to_csv(os.path.join(dirname, "scaled_dataframe.csv"), index=False)

        self.reference_data.to_csv(os.path.join(dirname, "reference_data.csv"), index=False)

    @classmethod
    def load(cls, dirname:str):
        """
        load the dataset by the directory

        Parameters
        ----------
        dirname : str
            Directory path to load the dataset
            
        Returns
        -------
        dataset: BaseDataset
            loaded dataset
        
        Example
        -------
        >>> dataset = BaseDataset.load("../demo_result/dataset/")
        """
        if not os.path.exists(dirname):
            raise ValueError("dir is not exists")

        with open(os.path.join(dirname, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)


        scale_info = joblib.load(os.path.join(dirname, "scale_info.joblib"))

        dataframe = pd.read_csv(os.path.join(dirname, "dataframe.csv"))
        scaled_dataframe = pd.read_csv(os.path.join(dirname, "scaled_dataframe.csv"))
        reference_data = pd.read_csv(os.path.join(dirname, "reference_data.csv"))
        dataset = cls(
            data = dataframe,
            x_columns = metadata.get("x_columns", None),
            y_column = metadata.get("y_column", None),
            id_column = metadata.get("id_column", None),
        )

        dataset.scalers = scale_info
        dataset.scaled_dataframe = scaled_dataframe
        dataset.network_input = np.load(
            os.path.join(dirname, "network_input.npy")).astype(np.float32)
        dataset.transform_x_y(scale_info.get("x",None),scale_info.get("y",None))
        dataset.reference_data = reference_data
        return dataset

class GTNNWRDataset(BaseDataset):
    """
    GTNNWR dataset is used to train the GTNNWR model.
    It inherits from BaseDataset and modifies the network_input attribute.

    Parameters
    ----------
    data: pandas.DataFrame
        dataframe
    x_columns: list
        independent variable column name
    y_column: list
        dependent variable column name
    id_column: list
        sample identifier column name
    """
    def __init__(self, data, x_columns, y_column, id_column=None):
        super().__init__(data, x_columns, y_column, id_column)
    
    def transform_network_input(self, network_input_scaler = None):
        r"""
        Scale the network input using the specified scaler
        
        Parameters
        ----------
        network_input_scaler: object
            scaler of network_input
        """
        if network_input_scaler is not None:
            self.network_input = network_input_scaler.transform(self.network_input.reshape(-1, 2)).reshape(self.network_input.shape)
            self.scalers["network_input"] = network_input_scaler
    
    def inverse_transform_network_input(self, network_input = None):
        r"""
        Inverse transform the network input using the specified scaler
        
        Parameters
        ----------
        network_input: np.ndarray
            network input to be inverse transformed
        """
        if self.scalers.get("network_input", None) is not None:
            network_input = self.scalers["network_input"].inverse_transform(network_input.reshape(-1, 2)).reshape(network_input.shape)
        return network_input

def create_predict_dataset(base_class: BaseDataset, **kwargs):
    r"""
    Create a predict dataset by the `base_class`.
    
    Parameters
    ----------
    base_class: BaseDataset
        base dataset class
    **kwargs: dict
        other parameters
        * data: pandas.DataFrame
            dataframe
        * x_columns: list
            independent variable column name
        * id_column: list
            sample identifier column name
    
    Returns
    -------
    dataset: PredictDataset(base_class)
    
    Example
    -------
    >>> predict_dataset = create_predict_dataset(
    ...     base_class = GTNNWRDataset,
    ...     data = dataframe,
    ...     x_columns = x_columns,
    ...     id_column = id_column,
    ... )
    """
    data = kwargs.get("data", None)
    x_columns = kwargs.get("x_columns", None)
    id_column = kwargs.get("id_column", None)
    class PredictDataset(base_class):
        """
        Predict dataset is used to predict the dependent variable of the data.

        Parameters
        ----------
        data: pandas.DataFrame
            dataframe
        x_columns: list
            independent variable column name
        id_column: list
            sample identifier column name
        """
        def __init__(self, data, x_columns, id_column=None):
            if id_column is None:
                id_column = "__idx__"
                data[id_column] = np.arange(len(data))
                
                warnings.warn(
                    "id_column is None, use default id_column '__idx__'",
                    UserWarning,
                )
            super().__init__(data, x_columns, None, [id_column])

        def __getitem__(self, index):
            r"""
            Retrieve a sample from the dataset by index, 
            returning all data components needed for model prediction.

            Parameters
            ----------
            index: int
                the index of sample

            Returns
            -------
            network_input: torch.Tensor
                spatial (and temporal) distance matrix
            x_tensor: torch.Tensor
                input features
            id_tensor: torch.Tensor
                sample identifiers
            """

            if self.kwargs['bias']:
                x_tensor = torch.cat([
                    torch.tensor(self.x_data[index], dtype=torch.float),
                    torch.ones(1, dtype=torch.float)
                ], dim=0)
            else:
                x_tensor = torch.tensor(self.x_data[index], dtype=torch.float)
            id_tensor = torch.tensor(self.id_data[index], dtype=torch.float)
            network_input = torch.tensor(self.network_input[index], dtype=torch.float)

            return network_input, x_tensor, id_tensor
    return PredictDataset(data, x_columns, id_column)


def init_dataset(data,
                 test_ratio,
                 valid_ratio,
                 x_columns,
                 y_column,
                 spatial_column=None,
                 temporal_column=None,
                 id_column=None,
                 **kwargs
                 ):
    r"""
    Initialize the dataset and return the training set, validation set, and test set for the model.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to be initialized.
    test_ratio : float
        The ratio of test data, which is the ratio of the entire dataset.
    valid_ratio : float
        The ratio of validation data, which is the ratio of the training set.
    x_columns : list
        The name of the input attribute column.
    y_column : list
        The name of the label column.
    spatial_column : list
        The name of the spatial attribute column.
    temporal_column : list
        The name of the temporal attribute column.
    id_column : list
        The name of the ID column.
    **kwargs: dict
        Additional keyword arguments for initializing the dataset.
        * sample_seed: int
            The random seed for sampling.
        * process_fun: callable
            The data pre-processing function, e.g., "minmax_scale", "standard_scale".
        * process_var: list
            The variable to be processed, which can be ["x"], ["y"], or ["x", "y"].
        * model_type: str
            The model to be used, e.g., "gnnwr", "gtnnwr".
        * spatial_fun: callable
            The function for calculating spatial distance.
        * temporal_fun: callable
            The function for calculating temporal distance.
        * cv_start_idx: int
            The start index for cross-validation.
        * reference_data : Union[str, pandas.DataFrame]
            Reference points for calculating the distance. 
            It can be a string ["train", "train_val"] or a pandas DataFrame.
        * dropna: bool
            Whether to drop rows with missing values.
            
    Returns
    -------
    train_dataset : BaseDataset
        The training dataset.
    valid_dataset : BaseDataset
        The validation dataset.
    test_dataset : BaseDataset
        The test dataset.

    Example
    -------
    >>> train_set, val_set, test_set = init_dataset(
    ...     data,
    ...     test_ratio=0.2,
    ...     valid_ratio=0.2,
    ...     x_columns=['x1', 'x2'],
    ...     y_column=['y'],
    ...     spatial_column=['lat', 'lng'],
    ...     temporal_column=['time'],
    ...     id_column=['id'],
    ...     sample_seed=42,
    ...     process_fun="minmax_scale",
    ...     process_var=["x"],
    ...     model_type="gnnwr",
    ...     spatial_fun=BasicDistance,
    ...     temporal_fun=ManhattanDistance,
    ...     cv_start_idx=0,
    ...     reference_data="train",
    ...     dropna=True,
    ... )
    """

    sample_seed = kwargs.get("sample_seed", 42)
    process_fun = kwargs.get("process_fun", "minmax_scale")
    process_var = kwargs.get("process_var", None)
    model_type = kwargs.get("model_type", "gnnwr")
    spatial_fun = kwargs.get("spatial_fun", BasicDistance)
    temporal_fun = kwargs.get("temporal_fun", ManhattanDistance)
    cv_start_idx = kwargs.get("cv_start_idx", 0)
    reference_data = kwargs.get("reference_data", None)
    dropna = kwargs.get("dropna", True)

    if spatial_fun is None:
        # if dist_fun is None, raise error
        raise ValueError(
            "dist_fun must be a function that can process the data")

    if spatial_column is None:
        # if dist_column is None, raise error
        raise ValueError(
            "dist_column must be a column name in data")
    if dropna:
        orilen = data.shape[0]
        data.dropna(axis=0, how='any', inplace=True)
        if orilen > data.shape[0]:
            warnings.warn(
                f"Dropping {orilen - data.shape[0]} {'' if orilen - data.shape[0] == 1 else 'rows'} with missing values. To forbid dropping, you need to set the argument dropna=False")
    if id_column is None:
        id_column = ['id']
        if 'id' not in data.columns:
            data['id'] = np.arange(len(data))
        else:
            warnings.warn("id_column is None and use default id column in data", RuntimeWarning)
    process_var = ["x"] if process_var is None else process_var
    np.random.seed(sample_seed) # pylint: disable=no-member
    data = data.sample(frac=1)  # shuffle data
    # data split
    test_data = data[int((1 - test_ratio) * len(data)):]
    train_data = data[:int((1 - test_ratio) * len(data))]
    val_data = train_data[
               int(cv_start_idx * valid_ratio * len(train_data)):int((1 + cv_start_idx) * valid_ratio * len(train_data))]
    train_data = pd.concat([train_data[:int(cv_start_idx * valid_ratio * len(train_data))],
                                train_data[int((1 + cv_start_idx) * valid_ratio * len(train_data)):]])
    return init_dataset_split(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        x_columns=x_columns,
        y_column=y_column,
        spatial_column=spatial_column,
        temporal_column=temporal_column,
        id_column=id_column,
        process_fun=process_fun,
        process_var = process_var,
        model_type=model_type,
        spatial_fun=spatial_fun,
        temporal_fun=temporal_fun,
        reference_data=reference_data,
        dropna=dropna
    )

def init_dataset_split(train_data,
                       val_data,
                       test_data,
                       x_columns,
                       y_column,
                       spatial_column=None,
                       temporal_column=None,
                       id_column=None,
                       **kwargs
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
    x_columns : list
        The name of the input attribute column.
    y_column : list
        The name of the label column.
    spatial_column : list
        The name of the spatial attribute column.
    temporal_column : list
        The name of the temporal attribute column.
    id_column : list
        The name of the ID column.
    **kwargs
        Additional keyword arguments for initializing the dataset.
        * process_fun : callable
            The data pre-processing function.
        * process_var : list
            The name of the attribute column to be processed.
        * model_type : str
            The model to be used, e.g., "gnnwr", "gtnnwr".
        * spatial_fun : callable
            The function for calculating spatial distance.
        * temporal_fun : callable
            The function for calculating temporal distance.
        * reference_data : Union[str, pandas.DataFrame]
            Reference points for calculating the distance. 
            It can be a string ["train", "train_val"] or a pandas DataFrame.
        * dropna : bool
            A flag indicating whether to drop NaN values.

    Returns
    -------
    train_dataset : BaseDataset
        The training dataset.
    valid_dataset : BaseDataset
        The validation dataset.
    test_dataset : BaseDataset
        The test dataset.
    """
    process_fun = kwargs.get("process_fun", "minmax_scale")
    process_var = kwargs.get("process_var", ["x"])
    model_type = kwargs.get("model_type", "gnnwr")
    spatial_fun = kwargs.get("spatial_fun", BasicDistance)
    temporal_fun = kwargs.get("temporal_fun", ManhattanDistance)
    reference_data = kwargs.get("reference_data", None)
    dropna = kwargs.get("dropna", True)

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
        orilen_train = train_data.shape[0]
        train_data = train_data.dropna(axis=0)

        if orilen_train > train_data.shape[0]:
            warnings.warn(
                f"Dropping {orilen_train - train_data.shape[0]} {'row' if orilen_train - train_data.shape[0] == 1 else 'rows'}\
                         with missing values. To forbid dropping, you need to set the argument dropna=False")
        # val_data
        orilen_val = val_data.shape[0]
        val_data = val_data.dropna(axis=0)
        if orilen_val > val_data.shape[0]:
            warnings.warn(
                f"Dropping {orilen_val - val_data.shape[0]} {'row' if orilen_val - val_data.shape[0] == 1 else 'rows'}\
                         with missing values. To forbid dropping, you need to set the argument dropna=False")  
        # test_data
        orilen_test = test_data.shape[0]
        test_data = test_data.dropna(axis=0)
        if orilen_test > test_data.shape[0]:
            warnings.warn(
                f"Dropping {orilen_test - test_data.shape[0]} {'row' if orilen_test - test_data.shape[0] == 1 else 'rows'}\
                         with missing values. To forbid dropping, you need to set the argument dropna=False")  
    
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

    # set Reference
    if reference_data is None:
        reference_data = train_data
    elif isinstance(reference_data, str):
        if reference_data == "train":
            reference_data = train_data
        elif reference_data == "train_val":
            reference_data = pd.concat([train_data, val_data])
        else:
            raise ValueError("reference_data str must be 'train' or 'train_val'")

    if not isinstance(reference_data, pd.DataFrame):
        raise ValueError("reference_data must be a pandas.DataFrame")
    if not all(col in reference_data.columns for col in spatial_column):
        raise ValueError(f"spatial_column {spatial_column} not in reference_data columns {reference_data.columns}")
    
    # data pre-process
    scaler_x = None
    scaler_y = None
    if process_fun == "minmax_scale":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif process_fun == "standard_scale":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError("process_fun must be 'minmax_scale' or 'standard_scale'")
    if "x" in process_var:
        scaler_x.fit(train_data[x_columns].values)
    else:
        scaler_x = None
    if "y" in process_var:
        scaler_y.fit(train_data[y_column].values)
    else:
        scaler_y = None
    
    if model_type == "gnnwr":
        train_dataset, val_dataset, test_dataset = _init_gnnwr_dataset(
            train_data, val_data, test_data,
            x_columns, y_column, id_column,
            process_fun,
            process_var,
            spatial_column,
            spatial_fun,
            reference_data,
        )
    elif model_type == "gtnnwr":
        assert temporal_column is not None, "temporal_column must be not None in gtnnwr"
        train_dataset, val_dataset, test_dataset = _init_gtnnwr_dataset(
            train_data, val_data, test_data,
            x_columns, y_column, id_column,
            process_fun,
            process_var,
            spatial_column,
            spatial_fun,
            temporal_column,
            temporal_fun,
            reference_data,
        )
    else:
        raise ValueError(f"model_type {model_type} not supported")
    # Other model init methods can be added here.

    # record reference
    train_dataset.reference_data = reference_data
    val_dataset.reference_data = reference_data
    test_dataset.reference_data = reference_data
    return train_dataset, val_dataset, test_dataset

def init_dataset_cv(data,
                    test_ratio,
                    k_fold,
                    x_columns,
                    y_column,
                    spatial_column=None,
                    temporal_column=None,
                    id_column=None,
                    **kwargs):
    r"""
    Initialize dataset for cross validation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    test_ratio : float
        Test set ratio.
    k_fold : int
        K of k-fold.
    x_columns : list
        The name of the input attribute column.
    y_column : str
        The name of the label column.
    spatial_column : list, optional
        The name of the spatial attribute column.
    temporal_column : list, optional
        The name of the temporal attribute column.
    id_column : list, optional
        The name of the ID column.
    **kwargs : dict
        Additional keyword arguments for initializing the dataset.
        look up the init_dataset function for more details.
        
    Returns
    -------
    cv_data_set : list
        List of cross validation datasets.
    test_dataset : BaseDataset
        Test dataset.
    """
    cv_data_set = []
    valid_ratio = (1 - test_ratio) / k_fold
    test_dataset = None
    for _ in range(k_fold):
        train_dataset, val_dataset, test_dataset = init_dataset(data, 
                                                                test_ratio, 
                                                                valid_ratio, 
                                                                x_columns, 
                                                                y_column,
                                                                spatial_column,
                                                                temporal_column,
                                                                id_column,
                                                                kwargs=kwargs)
        cv_data_set.append((train_dataset, val_dataset))
    return cv_data_set, test_dataset


def init_predict_dataset(data : pd.DataFrame,
                         x_columns: list,
                         spatial_columns: list=None,
                         id_column: list=None,
                         temporal_columns: list=None,
                         scalers: dict=None,
                         reference_data: pd.DataFrame=None,
                         model_type: str="gnnwr",
                         spatial_fun=BasicDistance,
                         temporal_fun=ManhattanDistance):
    """
    initialize predict dataset

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    x_columns : list
        The name of the input attribute column.
    spatial_columns : list, optional
        The name of the spatial attribute column.
    id_column : list, optional
        The name of the ID column.
    temporal_columns : list, optional
        The name of the temporal attribute column.
    scalers : dict, optional
        Scalers for data processing.
    reference_data : pandas.DataFrame, optional
        Reference data.
    model_type : str, optional
        Model type, by default `gnnwr`.
    spatial_fun : function, optional
        Spatial distance calculate function, by default `BasicDistance`.
    temporal_fun : function, optional
        Temporal distance calculate function, by default `ManhattanDistance`.
    """

    if spatial_fun is None:
        # if dist_fun is None, raise error
        raise ValueError(
            "dist_fun must be a function that can process the data")

    if spatial_columns is None:
        # if dist_column is None, raise error
        raise ValueError(
            "dist_column must be a column name in data")

    if model_type == "gnnwr":
        predict_dataset = _init_gnnwr_predict_dataset(
            data, x_columns, id_column, spatial_columns, scalers, reference_data, spatial_fun
        )

    elif model_type == "gtnnwr":
        assert temporal_columns is not None, "temporal_column must be not None in gtnnwr"
        predict_dataset = _init_gtnnwr_predict_dataset(
            data, x_columns, id_column, spatial_columns, temporal_columns, scalers, reference_data, spatial_fun, temporal_fun
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return predict_dataset

def _init_gnnwr_dataset(
    train_data, val_data, test_data,
    x_columns, y_column, id_column,
    process_fun="minmax_scale",
    process_var=None,
    spatial_column=None,
    spatial_fun=BasicDistance,
    reference_data=None
):
    r"""
    initialize gnnwr dataset

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data.
    val_data : pd.DataFrame
        Validation data.
    test_data : pd.DataFrame
        Test data.
    x_columns : list
        The name of the input attribute column.
    y_column : str
        The name of the label column.
    id_column : str
        The name of the ID column.
    process_fun : str
        Data process function, by default "minmax_scale".
    process_var : list
        Variables to process, by default ["x"].
    spatial_column : list
        Spatial distance column name.
    spatial_fun : function
        Spatial distance calculate function, by default BasicDistance.
    reference_data : pd.DataFrame
        Reference data, by default train_data.
    """
    train_dataset = BaseDataset(train_data, x_columns, y_column, id_column)
    val_dataset = BaseDataset(val_data, x_columns, y_column, id_column)
    test_dataset = BaseDataset(test_data, x_columns, y_column, id_column)
    if process_var is None:
        process_var = ["x"]
        
    train_dataset.network_input = _init_gnnwr_distance(
        reference_data[spatial_column].values,
        train_dataset.dataframe[spatial_column].values,
        spatial_fun
    )
    val_dataset.network_input = _init_gnnwr_distance(
        reference_data[spatial_column].values,
        val_dataset.dataframe[spatial_column].values,
        spatial_fun
    )
    test_dataset.network_input = _init_gnnwr_distance(
        reference_data[spatial_column].values,
        test_dataset.dataframe[spatial_column].values,
        spatial_fun
    )

    if process_fun == "minmax_scale":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_network = MinMaxScaler()
    elif process_fun == "standard_scale":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_network = StandardScaler()
    else:
        scaler_x = None
        scaler_y = None
        scaler_network = None

    scaler_x = None if "x" not in process_var else scaler_x
    scaler_y = None if "y" not in process_var else scaler_y

    if scaler_x is not None:
        scaler_x.fit_transform(train_dataset.x_data)
    if scaler_y is not None:
        scaler_y.fit_transform(train_dataset.y_data)
    if scaler_network is not None:
        flatten_network_input = train_dataset.network_input.reshape(-1, 1)
        flatten_network_input = scaler_network.fit_transform(flatten_network_input)

    train_dataset.transform_x_y(scaler_x, scaler_y)
    train_dataset.transform_network_input(scaler_network)
    val_dataset.transform_x_y(scaler_x, scaler_y)
    val_dataset.transform_network_input(scaler_network)
    test_dataset.transform_x_y(scaler_x, scaler_y)
    test_dataset.transform_network_input(scaler_network)
    
    return train_dataset, val_dataset, test_dataset

def _init_gnnwr_predict_dataset(
        data,
        x_columns,
        id_column,
        spatial_columns,
        scalers,
        reference_data,
        spatial_fun):
    r"""
    Initialize GNNWR model's predict dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input predict data.
    x_columns : list
        The name of the input attribute column.
    id_column : list
        The name of the ID column.
    spatial_columns : list
        The name of the spatial attribute column.
    scalers : dict
        Data scalers dictionary, containing 'x' and 'network' keys.
    reference_data : pd.DataFrame
        Reference data, used for calculating spatial distance.
    spatial_fun : function
        Spatial distance calculate function.
        
    Returns
    -------
    predict_dataset : PredictDataset
        Initialized predict dataset.
    """
    predict_dataset = create_predict_dataset(BaseDataset, data=data, x_columns=x_columns, id_column=id_column)
    predict_dataset.network_input = _init_gnnwr_distance(
        reference_data[spatial_columns].values, data[spatial_columns].values, spatial_fun
    )
    if scalers is not None:
        predict_dataset.transform_x_y(scalers.get("x", None), None)
        predict_dataset.transform_network_input(scalers.get("network", None))
    
    return predict_dataset

def _init_gnnwr_distance(refer_data, data, spatial_fun=BasicDistance):
    r"""
    Calculate the spatial distance between points in data and Reference points.
    
    Parameters
    ----------
    refer_data : numpy.nDarray
        Reference points for calculating the distance.
    data : numpy.nDarray
        The data subset used for calculating the distance.
    spatial_fun : function
        The function used for calculating the distance. Default is `BasicDistance`.

    Returns
    -------
    spatial_distance : numpy.nDarray
        matrix of spatial distance between points in data and Reference points
    """

    spatial_distance = spatial_fun(data, refer_data)
    return spatial_distance

def _init_gtnnwr_dataset(
            train_data, val_data, test_data,
            x_columns, y_column, id_column,
            process_fun,
            process_var,
            spatial_column,
            spatial_fun,
            temporal_column,
            temporal_fun,
            reference_data,
        ):
    r"""
    Create a GTNNWR dataset and preprocess the data and the network input.
    
    Parameters
    ----------
    train_data : pandas.DataFrame
        The training data.
    val_data : pandas.DataFrame
        The validation data.
    test_data : pandas.DataFrame
        The test data.
    x_columns : list of str
        The name of the input attribute column.
    y_column : str
        The name of the label column.
    id_column : str
        The name of the ID column.
    process_fun : str
        The function used for preprocessing.
    process_var : list of str
        The variables to be processed.
    spatial_column : str
        The name of the spatial attribute column.
    spatial_fun : function
        The function used for calculating the spatial distance.
    temporal_column : str
        The name of the temporal attribute column.
    temporal_fun : function
        The function used for calculating the temporal distance.
    reference_data : pandas.DataFrame
        The reference data used for calculating the distance.
    
    Returns
    -------
    train_dataset : GTNNWRDataset
        The training dataset.
    val_dataset : GTNNWRDataset
        The validation dataset.
    test_dataset : GTNNWRDataset
        The test dataset.
    """
    train_dataset = GTNNWRDataset(train_data, x_columns, y_column, id_column)
    val_dataset = GTNNWRDataset(val_data, x_columns, y_column, id_column)
    test_dataset = GTNNWRDataset(test_data, x_columns, y_column, id_column)
    train_dataset.network_input = _init_gtnnwr_distance(
        [reference_data[spatial_column].values,reference_data[temporal_column].values],
        [train_dataset.dataframe[spatial_column].values,train_dataset.dataframe[temporal_column].values],
        spatial_fun,
        temporal_fun,
    )
    val_dataset.network_input = _init_gtnnwr_distance(
        [reference_data[spatial_column].values,reference_data[temporal_column].values],
        [val_dataset.dataframe[spatial_column].values,val_dataset.dataframe[temporal_column].values],
        spatial_fun,
        temporal_fun,
    )
    test_dataset.network_input = _init_gtnnwr_distance(
        [reference_data[spatial_column].values,reference_data[temporal_column].values],
        [test_dataset.dataframe[spatial_column].values,test_dataset.dataframe[temporal_column].values],
        spatial_fun,
        temporal_fun,
    )
    if process_fun == "minmax_scale":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_network = MinMaxScaler()
    elif process_fun == "standard_scale":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_network = StandardScaler()
    else:
        scaler_x = None
        scaler_y = None
        scaler_network = None

    scaler_x = None if "x" not in process_var else scaler_x
    scaler_y = None if "y" not in process_var else scaler_y

    if scaler_x is not None:
        scaler_x.fit_transform(train_dataset.x_data)
    if scaler_y is not None:
        scaler_y.fit_transform(train_dataset.y_data)
    if scaler_network is not None:
        flatten_network_input = train_dataset.network_input.reshape(-1, 2)
        flatten_network_input = scaler_network.fit_transform(flatten_network_input)

    train_dataset.transform_x_y(scaler_x, scaler_y)
    train_dataset.transform_network_input(scaler_network)
    val_dataset.transform_x_y(scaler_x, scaler_y)
    val_dataset.transform_network_input(scaler_network)
    test_dataset.transform_x_y(scaler_x, scaler_y)
    test_dataset.transform_network_input(scaler_network)

    return train_dataset, val_dataset, test_dataset

def _init_gtnnwr_predict_dataset(
            data,
            x_columns,
            id_column,
            spatial_column,
            temporal_column,
            scalers,
            reference_data,
            spatial_fun=BasicDistance,
            temporal_fun=ManhattanDistance):
    r"""
    Create a predict dataset for GTNNWR, and preprocess the data if scalers are provided.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The data subset used for prediction.
    x_columns : list of str
        The name of the input attribute column.
    id_column : str
        The name of the ID column.
    spatial_column : str
        The name of the spatial attribute column.
    temporal_column : str
        The name of the temporal attribute column.
    scalers : dict
        The scalers used for preprocessing.
    reference_data : pandas.DataFrame
        The reference data used for calculating the distance.
    spatial_fun : function
        The function used for calculating the spatial distance.
    temporal_fun : function
        The function used for calculating the temporal distance.
    """
    predict_dataset = create_predict_dataset(GTNNWRDataset, data=data, x_columns=x_columns, id_column=id_column)
    predict_dataset.network_input = _init_gtnnwr_distance(
        [reference_data[spatial_column].values,reference_data[temporal_column].values],
        [predict_dataset.dataframe[spatial_column].values,predict_dataset.dataframe[temporal_column].values],
        spatial_fun,
        temporal_fun,
    )
    if scalers is not None:
        predict_dataset.transform_x_y(scalers.get("x", None), None)
        predict_dataset.transform_network_input(scalers.get("network", None))

    return predict_dataset

def _init_gtnnwr_distance(refer_data,
                          data,
                          spatial_fun=BasicDistance,
                          temporal_fun=ManhattanDistance):
    r"""
    Parameters
    ----------
    refer_data : numpy.nDarray
        Reference points for calculating the distance.
    data : numpy.nDarray
        The data subset used for calculating the distance.
    spatial_fun : function
        The function used for calculating the spatial distance. Default is `BasicDistance`.
    temporal_fun : function
        The function used for calculating the temporal distance. Default is `ManhattanDistance`.

    Returns
    -------
    distance_matrix: numpy.nDarray
        Matrix of distance between points in data and Reference points. 
        The shape is (n_data, n_refer, 2), where the 
        first dimension is the number of data points, 
        the second dimension is the number of reference points, and the 
        third dimension is the spatial and temporal distance respectively.
    """
    refer_s_distance, refer_t_distance = refer_data[0], refer_data[1]
    data_s_distance, data_t_distance = data[0], data[1]

    data_s_distance = spatial_fun(data_s_distance, refer_s_distance)
    data_t_distance = temporal_fun(data_t_distance, refer_t_distance)

    return np.stack((data_s_distance, data_t_distance), axis=2)
