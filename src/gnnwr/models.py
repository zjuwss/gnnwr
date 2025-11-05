r"""
GNNWR Models Module

This module provides implementations of spatiotemporal intelligent regression models, including GNNWR (Geographically Neural Network Weighted Regression)
and GTNNWR (Geographically and Temporally Neural Network Weighted Regression).


References
----------
.. [1] `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`__
.. [2] `Geographically and temporally neural network weighted regression for modeling spatiotemporal non-stationary relationships <https://doi.org/10.1080/13658816.2020.1775836>`__
"""
import datetime
import os
import warnings
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from .networks import SWNN, STPNN
from .utils import OLS, DIAGNOSIS
from .datasets import BaseDataset

class GNNWR:
    r"""
    GNNWR(Geographically neural network weighted regression) is a model to address spatial non-stationarity in various domains with complex geographical processes,
    which comes from the paper `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`__.

    Parameters
    ----------
    train_dataset : BaseDataset
        the dataset of training
    valid_dataset : BaseDataset
        the dataset of validation
    test_dataset : BaseDataset
        the dataset of testing
    dense_layers : list
        the dense layers of the model (default: ``None``)
        Default structure is created by `default_dense_layers` function.
    start_lr : float
        the start learning rate of the model (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"AdamW"``)
        choose from "SGD","Adam","RMSprop","AdamW","Adadelta"
    model_name : str
        the name of the model (default: ``"GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")``)
    model_save_path : str
        the path of the model (default: ``"../gnnwr_models"``)
    write_path : str
        the path of the log (default: ``"../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")``)
    use_gpu : bool
        whether use gpu or not (default: ``True``)
    use_ols : bool
        whether use ols or not (default: ``True``)
    log_path : str
        the path of the log (default: ``"../gnnwr_logs/"``)
    log_file_name : str
        the name of the log (default: ``"gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"``)
    log_level : int
        the level of the log (default: ``logging.INFO``)
    optimizer_params : dict, optional
        Additional parameters for optimizer and learning rate scheduler
    tensorboard_mode : bool
        whether use tensorboard or not (default: ``True``)
    log_mode : bool
        whether use log or not (default: ``True``)
    kwargs : dict, optional
        Additional parameters for model, including:
        - drop_out : float
            the drop out rate of the model (default: ``0.2``)
        - batch_norm : bool, optional
            whether use batch normalization (default: ``True``)
        - activate_func : torch.nn
            the activate function of the model (default: ``nn.PReLU(init=0.4)``)
    """

    def __init__(
            self,
            train_dataset: BaseDataset,
            valid_dataset: BaseDataset,
            test_dataset: BaseDataset,
            dense_layers: list = None,
            start_lr: float = .1,
            optimizer: str = "AdamW",
            model_name="GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
            model_save_path="../gnnwr_models",
            write_path="../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            use_gpu: bool = True,
            use_ols: bool = True,
            log_path: str ="../gnnwr_logs",
            log_file_name="gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
            log_level=logging.INFO,
            optimizer_params=None,
            tensorboard_mode: bool = True,
            log_mode: bool = True,
            **kwargs
    ):
        self._train_dataset = train_dataset  # train dataset
        self._valid_dataset = valid_dataset  # valid dataset
        self._test_dataset = test_dataset  # test dataset

        
        self._insize = train_dataset.datasize  # size of input layer
        self._outsize = train_dataset.coefsize  # size of output layer

        self._dense_layers = dense_layers  # structure of layers
        self._start_lr = start_lr  # initial learning rate
        self._drop_out = kwargs.get("drop_out", 0.2)  # drop_out ratio
        self._batch_norm = kwargs.get("batch_norm", True)  # batch normalization
        self._activate_func = kwargs.get("activate_func", nn.PReLU(init=0.4))  # activate function , default: PRelu(0.4)

        self._model = SWNN(self._dense_layers, self._insize, self._outsize,
                           self._drop_out, self._activate_func, self._batch_norm)  # model

        self._coefficient = OLS(
            train_dataset.scaled_dataframe, train_dataset.x_columns, train_dataset.y_column).params  # coefficients of OLS

        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # layer to multiply OLS coefficients and model output
        if use_ols:
            self._out.weight = nn.Parameter(torch.tensor([self._coefficient]).to(
                torch.float32), requires_grad=False)  # define the weight
        else:
            self._coefficient = np.ones((1, self._outsize))
            self._out.weight = nn.Parameter(torch.tensor(np.ones((1, self._outsize))).to(torch.float32), requires_grad=False)  # define the weight

        self._criterion = nn.MSELoss()  # loss function

        self._train_loss_list = []  # record the loss in training process
        self._valid_loss_list = []  # record the loss in validation process
        self._best_performance = {'train_r2': float('-inf'), 'valid_r2': float('-inf')}
        # Model information
        self._model_name = model_name  # model name
        self._model_save_path = model_save_path  # model save path
        if not os.path.exists(self._model_save_path):
            os.makedirs(self._model_save_path)

        if tensorboard_mode:
            self._writer = SummaryWriter(write_path)  # summary writer
        else:
            self._writer = None

        self._log_mode = log_mode  # log mode
        self._log_path = log_path  # log path
        self._log_file_name = log_file_name  # log file
        self._log_level = log_level  # log level
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)
        self.__is_trained = False  # whether the model is trained

        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False
        self._device = torch.device('cuda') if self._use_gpu else torch.device('cpu')

        self._optimizer = None
        self._scheduler = None
        self._optimizer_name = None
        self._optimizer_params = optimizer_params
        self.init_optimizer(optimizer, optimizer_params)  # initialize the optimizer
        

    def init_optimizer(self, optimizer, optimizer_params=None):
        r"""
        Initialize the optimizer.

        This method sets up the optimizer and the learning rate scheduler with the given parameters.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer to use. This should be one of the supported optimizers.
        optimizer_params : dict, optional
            A dictionary containing parameters for the optimizer and scheduler. If not provided,
            default values will be used. The dictionary can contain the following keys:

            - **scheduler** \: *str*
            
            The type of learning rate scheduler to use. Valid options include ``Special``,
            ``Constant``, and ``MultiStepLR``.
            - **maxlr** \: *float*
                
                The maximum learning rate for the scheduler, for ``Special``.
            - **minlr** \: *float*
                
                The minimum learning rate for the scheduler, for ``Special``.
            - **upepoch** \: *int*
                
                The number of epochs until the maximum learning rate is reached, for ``Special``.
            - **decayepoch** \: *int*
                
                The epoch at which learning rate decay starts, for ``Special``.
            - **decayrate** \: *float*
                
                The rate at which the learning rate decays, for ``Special``.
            - **stop_change_epoch** \: *int*

                The epoch at which to stop adjusting the learning rate, for ``Special``.
            - **stop_lr** \: *float*

                The learning rate to stop at after the specified epoch, for ``Special``.
            - **scheduler_milestones** \: *list*

                The epochs at which to decay the learning rate for a ``MultiStepLR`` scheduler.
            - **scheduler_gamma** \: *float*

                The factor by which the learning rate is reduced for a ``MultiStepLR`` scheduler.
            - **weight_decay** \: *float*

                The weight decay factor for the optimizer.

        Raises
        ------
        ValueError
            If an unsupported optimizer or scheduler is specified.

        """
        # initialize the optimizer
        if optimizer_params is None:
            optimizer_params = {}
        weigth_decay = optimizer_params.get("weight_decay", 1e-3)
        
        if optimizer == "SGD":
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "Adam":
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "AdamW":
            self._optimizer = optim.AdamW(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "RMSprop":
            self._optimizer = optim.RMSprop(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "Adadelta":
            self._optimizer = optim.Adadelta(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        else:
            raise ValueError("Invalid Optimizer")
        self._optimizer_name = optimizer  # optimizer name

        # lr scheduler
        
        scheduler = optimizer_params.get("scheduler", "Constant")
        scheduler_milestones = optimizer_params.get(
            "scheduler_milestones", [100, 500, 1000, 2000])
        scheduler_gamma = optimizer_params.get("scheduler_gamma", 0.5)
        scheduler_T_max = optimizer_params.get("scheduler_T_max", 1000)
        scheduler_eta_min = optimizer_params.get("scheduler_eta_min", 0.01)
        scheduler_T_0 = optimizer_params.get("scheduler_T_0", 100)
        scheduler_T_mult = optimizer_params.get("scheduler_T_mult", 3)
        if scheduler == "MultiStepLR":
            self._scheduler = optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
        elif scheduler == "CosineAnnealingLR":
            self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=scheduler_T_max, eta_min=scheduler_eta_min)
        elif scheduler == "CosineAnnealingWarmRestarts":
            self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=scheduler_T_0, T_mult=scheduler_T_mult, eta_min=scheduler_eta_min)
        elif scheduler == "Special":
            maxlr = optimizer_params.get("maxlr", 0.1)
            minlr = optimizer_params.get("minlr", 0.01)
            upepoch = optimizer_params.get("upepoch", 10000)
            uprate = (maxlr - minlr) / upepoch
            decayepoch = optimizer_params.get("decayepoch", 20000)
            decayrate = optimizer_params.get("decayrate", 0.95)
            stop_change_epoch = optimizer_params.get("stop_change_epoch", 30000)
            stop_lr = optimizer_params.get("stop_lr", 0.001)
            lambda_lr = lambda epoch: epoch * uprate + minlr if epoch < upepoch else (
                maxlr if epoch < decayepoch else maxlr * (
                        decayrate ** ((epoch - decayepoch) // 200))) if epoch < stop_change_epoch else stop_lr
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda_lr)
        elif scheduler == "Constant":
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda epoch: 1)
        else:
            raise ValueError("Invalid Scheduler")

    def __train(self, dataloader):
        """
        train the network
        """
        self._model.train()  # set the model to train mode
        train_loss = 0  # initialize the loss
        weight_all = []
        x_true = []
        y_true = []
        y_pred = []
        for _, (data, coef, label, _) in enumerate(dataloader):
            # move the data to gpu
            data, coef, label = data.to(self._device), coef.to(self._device), label.to(self._device)

            self._optimizer.zero_grad()  # zero the gradient
            weight = self._model(data)
            weight_all.append(weight)
            output = self._out(weight.mul(coef))
            x_true.append(coef)
            y_true.append(label)
            y_pred.append(output)
            loss = self._criterion(output, label) # calculate the loss
            loss.backward()  # back propagation
            self._optimizer.step()  # update the parameters
            train_loss += loss.item() # accumulate the loss
        x_true = torch.concatenate(x_true, dim=0)
        y_true = torch.concatenate(y_true, dim=0)
        y_pred = torch.concatenate(y_pred, dim=0)
        weight_all = torch.concatenate(weight_all, dim=0)
        train_loss /= len(dataloader)  # calculate the average loss
        self._train_loss_list.append(train_loss)  # record the loss
        return train_loss, weight_all, x_true, y_true, y_pred


    def __evaluate(self, dataloader:DataLoader):
        """
        Evaluate the model performance on a given dataset.
        """
        self._model.eval()
        eval_loss = 0
        x_data = []
        y_data = []
        y_pred = []
        weight_all = []
        with torch.no_grad():
            for data, coef, label, _ in dataloader:
                data, coef, label = data.to(self._device), coef.to(self._device), label.to(self._device)
                
                weight = self._model(data)
                output = self._out(weight.mul(coef))
                
                x_data.append(coef)
                y_data.append(label)
                y_pred.append(output)
                weight_all.append(weight)
                loss = self._criterion(output, label)

                eval_loss += loss.item() # accumulate the loss
            
            eval_loss /= len(dataloader)
        x_data = torch.concatenate(x_data, dim=0)
        y_data = torch.concatenate(y_data, dim=0)
        y_pred = torch.concatenate(y_pred, dim=0)
        weight_all = torch.concatenate(weight_all, dim=0)
        return eval_loss,weight_all, x_data, y_data, y_pred

    def run(self, max_epoch=1, early_stop=-1,**kwargs):
        """
        train the model and validate the model

        Parameters
        ----------
        max_epoch : int
            the max epoch of the training (default: ``1``)
        early_stop : int
            if the model has not been updated for ``early_stop`` epochs, the training will stop (default: ``-1``)

            if ``early_stop`` is ``-1``, the training will not stop until the max epoch
        kwargs : dict, optional
            Additional parameters for training, including:
            - print_frequency : int
                the frequency of printing the information (default: ``50``)
                it will be deprecated in the future, the information will be shown in tqdm

            - show_detailed_info : bool
                if ``True``, the detailed information will be shown (default: ``True``)
                it will be deprecated in the future, the information will be shown in tqdm
        """
        if kwargs.get("print_frequency") is not None:
            warnings.warn("The parameter print_frequency is deprecated, the information will be shown in tqdm")
        if kwargs.get("show_detailed_info") is not None:
            warnings.warn("The parameter show_detailed_info is deprecated, the information will be shown in tqdm")
        batch_size = kwargs.get("batch_size", 64)
        # model selection method
        model_selection = kwargs.get("model_selection", "val")
        self.__is_trained = True
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        # create file to record the information
        if self._log_mode:
            if not os.path.exists(self._log_path):
                os.mkdir(self._log_path)
            file_str = self._log_path + "/" + self._log_file_name
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                filename=file_str, level=logging.INFO)
        train_Dataloader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        valid_Dataloader = DataLoader(self._valid_dataset, batch_size=batch_size, shuffle=False)
        best_last_epoch = 0
        with tqdm(range(max_epoch)) as pbar:
            for epoch in pbar:
                # train the network
                train_loss, weight_all_train, x_true_train, y_true_train, y_pred_train = self.__train(train_Dataloader)
                train_Diagnosis = DIAGNOSIS(weight_all_train, x_true_train, y_true_train, y_pred_train)
                # validate the network
                eval_loss, weight_all_valid, x_true_valid, y_true_valid, y_pred_valid = self.__evaluate(valid_Dataloader)
                valid_Diagnosis = DIAGNOSIS(weight_all_valid, x_true_valid, y_true_valid, y_pred_valid)

                # earlystop
                if early_stop != -1:
                    if valid_Diagnosis.R2() > self._best_performance['valid_r2']:
                        self._best_performance['valid_r2'] = valid_Diagnosis.R2()
                        self._best_performance['train_r2'] = train_Diagnosis.R2()
                        torch.save(self._model, self._model_save_path + '/' + self._model_name + ".pkl")
                        best_last_epoch = 0
                    else:
                        best_last_epoch += 1

                # out put the information
                pbar.set_postfix(
                    {'Train Loss': f"{train_loss:.4f}", 
                                  'Train R2': f"{train_Diagnosis.R2():.4f}",
                                  'Train RMSE': f"{train_Diagnosis.RMSE():.4f}",
                                  'Train AIC': f"{train_Diagnosis.AIC():.4f}",
                                  'Valid Loss': f"{eval_loss:.4f}",
                                  'Valid R2': f"{valid_Diagnosis.R2():.4f}",
                                  'Valid RMSE': f"{valid_Diagnosis.RMSE():.4f}",
                                  'Best Valid R2': f"{self._best_performance['valid_r2']:.4f}",
                                  'Learning Rate': self._optimizer.param_groups[0]['lr']}
                                )
                self._train_loss_list.append(train_loss)
                self._valid_loss_list.append(eval_loss)

                self._scheduler.step()  # update the learning rate
                # tensorboard
                if self._writer is not None:
                    self._writer.add_scalar('Training/Learning Rate', self._optimizer.param_groups[0]['lr'], epoch)
                    self._writer.add_scalar('Training/Loss', self._train_loss_list[-1], epoch)
                    self._writer.add_scalar('Training/R2', train_Diagnosis.R2(), epoch)
                    self._writer.add_scalar('Training/RMSE', train_Diagnosis.RMSE(), epoch)
                    self._writer.add_scalar('Training/AIC', train_Diagnosis.AIC(), epoch)
                    self._writer.add_scalar('Training/AICc', train_Diagnosis.AICc(), epoch)
                    self._writer.add_scalar('Validation/Loss', self._valid_loss_list[-1], epoch)
                    self._writer.add_scalar('Validation/R2', valid_Diagnosis.R2(), epoch)
                    self._writer.add_scalar('Validation/Best R2', self._best_performance['valid_r2'], epoch)

                # log output
                if self._log_mode:
                    log_str = "Epoch: " + str(epoch + 1) + \
                            "; Train Loss: " + str(self._train_loss_list[-1]) + \
                            "; Train R2: {:5f}".format(train_Diagnosis.R2()) + \
                            "; Train RMSE: {:5f}".format(train_Diagnosis.RMSE()) + \
                            "; Train AIC: {:5f}".format(train_Diagnosis.AIC()) + \
                            "; Train AICc: {:5f}".format(train_Diagnosis.AICc()) + \
                            "; Valid Loss: " + str(self._valid_loss_list[-1]) + \
                            "; Valid R2: {:5f}".format(valid_Diagnosis.R2()) + \
                            "; Valid RMSE: {:5f}".format(valid_Diagnosis.RMSE()) + \
                            "; Learning Rate: " + str(self._optimizer.param_groups[0]['lr'])
                    logging.info(log_str)
                if 0 < early_stop < best_last_epoch:  # stop when the model has not been updated for long time
                    print(f"Training stop! Model has not been improved for over {best_last_epoch} epochs.")
                    break
        torch.save(self._model, self._model_save_path + '/' + self._model_name + "_last.pkl")
        if model_selection == "val":
            self.load_model(self._model_save_path + '/' + self._model_name + ".pkl")
        elif model_selection == "last":
            self.load_model(self._model_save_path + '/' + self._model_name + "_last.pkl")

    def predict(self, dataset:BaseDataset, **kwargs):
        """
        predict the result of the dataset

        Parameters
        ----------
        dataset : BaseDataset
            the dataset to be predicted

        Returns
        -------
        dataframe
            the Pandas dataframe of the dataset with the predicted result
        """
        batch_size = kwargs.get("batch_size", 64)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if not self.__is_trained:
            print("WARNING! The model hasn't been trained or loaded!")
        self._model.eval()
        weight = []
        y_pred = []
        with torch.no_grad():
            for data, coef, _ in dataloader:
                data, coef = data.to(self._device), coef.to(self._device)
                weight_batch = self._model(data)
                weight.append(weight_batch)
                y_pred.append(self._out(weight_batch * coef))
        weight = torch.cat(weight, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.cpu().detach().numpy()
        coefficient = weight.cpu().detach().numpy() * self._coefficient
        dataset.dataframe['pred_result'] = y_pred
        _, dataset.dataframe['denormalized_pred_result'] = dataset.inverse_transform_x_y(None,y_pred)

        return dataset.dataframe, coefficient

    def load_model(self, path, use_dict=False, map_location=None):
        """
        load the model from the path

        Parameters
        ----------
        path : str
            the path of the model
        use_dict : bool
            whether the function use dict to load the model (default: ``False``)
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``
        """
        if use_dict:
            data = torch.load(path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location, weights_only=False)
        if self._use_gpu:
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        self._model_save_path = os.path.dirname(path)
        self._model_name = os.path.basename(path).split('/')[-1].split('.')[0]
        self.__is_trained = True
        self.result_data = self.coefficient_result()

    def gpumodel_to_cpu(self, path, save_path, use_model=True):
        """
        convert gpu model to cpu model

        Parameters
        ----------
        path : str
            the path of the model
        save_path : str
            the path of the new model
        use_model : bool
            whether use dict to load the model (default: ``True``)
        """
        if use_model:
            data = torch.load(path, map_location='cpu', weights_only=False).state_dict()
        else:
            data = torch.load(path, map_location='cpu', weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        torch.save(new_state_dict, save_path)

    def get_loss(self):
        """
        get network's loss

        Returns
        -------
        list
            the list of the loss in training process and validation process
        """
        return self._train_loss_list, self._valid_loss_list

    def add_graph(self):
        """
        add the graph of the model to tensorboard
        """
        if self._writer is None:
            raise Exception("Tensorboard is not enabled!")
        train_Dataloader = DataLoader(self._train_dataset, batch_size=1, shuffle=False)
        for data, _, _, _ in train_Dataloader:
            if self._use_gpu:
                data = data.cuda()
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
                data = data.cpu()
            self._writer.add_graph(self._model, data)
            break
        print("Add Graph Successfully")

    def result(self, path=None, use_dict=False, map_location=None,**kwargs):
        """
        print the result of the model, including the model name, regression fomula and the result of test dataset

        Parameters
        ----------
        path : str
            the path of the model(default: ``None``)
            | if ``path`` is ``None``, the model will be loaded from ``self._modelSavePath + "/" + self._modelName + ".pkl"``
        use_dict : bool
            whether the function use dict to load the model (default: ``False``)
            | if ``use_dict`` is ``True``, the model will be loaded from ``path`` as dict
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``
        """
        # load model
        if not self.__is_trained:
            raise Exception("The model hasn't been trained or loaded!")
        if path is None:
            path = self._model_save_path + "/" + self._model_name + ".pkl"
        if use_dict:
            data = torch.load(path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location, weights_only=False)
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        batch_size = kwargs.get("batch_size", min(len(self._train_dataset), len(self._valid_dataset), len(self._test_dataset)))
        train_Dataloader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=False)
        valid_Dataloader = DataLoader(self._valid_dataset, batch_size=batch_size, shuffle=False)
        test_Dataloader = DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            _ , train_weight,train_x_true,train_y_true,train_y_pred = self.__evaluate(train_Dataloader)
            train_Diagnosis = DIAGNOSIS(train_weight,train_x_true,train_y_true,train_y_pred)
            _ , valid_weight,valid_x_true,valid_y_true,valid_y_pred = self.__evaluate(valid_Dataloader)
            valid_Diagnosis = DIAGNOSIS(valid_weight,valid_x_true,valid_y_true,valid_y_pred)
            test_loss , test_weight,test_x_true,test_y_true,test_y_pred = self.__evaluate(test_Dataloader)
            test_Diagnosis = DIAGNOSIS(test_weight,test_x_true,test_y_true,test_y_pred)

        if self._log_mode:
            logging.info(f"Test Loss: {test_loss:.5f}; Test R2: {test_Diagnosis.R2():.5f}")
        # print result
        # basic information
        print("--------------------Model Information-----------------")
        print("Model Name:           |", self._model_name)
        print("independent variable: |", self._train_dataset.x_columns)
        print("dependent variable:   |", self._train_dataset.y_column)
        # OLS
        print("\nOLS coefficients: ")
        for i in range(len(self._coefficient)):
            if i == len(self._coefficient) - 1:
                print("Intercept: {:.5f}".format(self._coefficient[i]))
            else:
                print("x{}: {:.5f}".format(i, self._coefficient[i]))
        print("\n--------------------Result Information----------------")
        print(f"Test Loss: | {test_loss:>25.5f}")
        print(f"Test R2  : | {test_Diagnosis.R2():>25.5f}")
        print(f"Train R2 : | {train_Diagnosis.R2():>25.5f}")
        print(f"Valid R2 : | {valid_Diagnosis.R2():>25.5f}")
        print(f"RMSE: | {test_Diagnosis.RMSE():>30.5f}")
        print(f"AIC:  | {test_Diagnosis.AIC():>30.5f}")
        print(f"AICc: | {test_Diagnosis.AICc():>30.5f}")
        print(f"F1:   | {test_Diagnosis.F1_Global():>30.5f}")
        print(f"F2:   | {test_Diagnosis.F2_Global():>30.5f}")
        F3_Local_dict = test_Diagnosis.F3_Local()[0]
        for key in F3_Local_dict:
            width = 30 - (len(key) - 4)
            print(f"{key}: | {F3_Local_dict[key]:>{width}.5f}")

    def reg_result(self, filename=None, model_path=None, **kwargs):
        """
        save the regression result of the model, including the coefficient of each argument, the bias and the predicted result

        Parameters
        ----------
        filename : str
            the path of the result file (default: ``None``)
            | if ``filename`` is ``None``, the result will not be saved as file
        model_path : str
            the path of the model (default: ``None``)
            | if ``model_path`` is ``None``, the model will be loaded from ``self._modelSavePath + "/" + self._modelName + ".pkl"``
        use_dict : bool
            whether use dict to load the model (default: ``False``)
            | if ``use_dict`` is ``True``, the model will be loaded from ``model_path`` as dict
        only_return : bool
            whether only return the result (default: ``False``)
            | if ``only_return`` is ``True``, the result will not be saved as file
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``

        Returns
        -------
        dataframe
            the Pandas dataframe of the result
        """
        if model_path is None:
            model_path = self._model_save_path + "/" + self._model_name + ".pkl"
            
        use_dict = kwargs.get("use_dict", False)
        map_location = kwargs.get("map_location", None)
        
        if use_dict:
            data = torch.load(model_path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(model_path, map_location=map_location, weights_only=False)

        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)
            self._model,self._out = self._model.cuda(),self._out.cuda()
        else:
            self._model, self._out = self._model.cpu(), self._out.cpu()
        
        batch_size = kwargs.get("batch_size", min(len(self._train_dataset), len(self._valid_dataset), len(self._test_dataset)))
        train_Dataloader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=False)
        valid_Dataloader = DataLoader(self._valid_dataset, batch_size=batch_size, shuffle=False)
        test_Dataloader = DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False)
      
        with torch.no_grad():
            # calculate the result of train dataset
            _, train_weight, _, _, train_y_pred = self.__evaluate(train_Dataloader)
            # calculate the result of valid dataset
            _, valid_weight, _, _, valid_y_pred = self.__evaluate(valid_Dataloader)
            # calculate the result of test dataset
            _, test_weight, _, _, test_y_pred = self.__evaluate(test_Dataloader)
        weight = torch.cat([train_weight, valid_weight, test_weight], dim=0).cpu().detach().numpy()
        y_pred = torch.cat([train_y_pred, valid_y_pred, test_y_pred], dim=0).cpu().detach().numpy()
        coef = weight * self._coefficient
        id_data = np.concatenate([self._train_dataset.id_data, self._valid_dataset.id_data, self._test_dataset.id_data], axis=0)
        result = np.concatenate([coef, y_pred, id_data.reshape(-1, 1)], axis=1)
        columns = list(self._train_dataset.x_columns)
        for idx,column in enumerate(columns):
            columns[idx] = "coef_" + column
        columns.append("bias")
        columns = columns + ["Pred_" + self._train_dataset.y_column[0]] + self._train_dataset.id_column
        result = pd.DataFrame(result, columns=columns)

        # set dataset belong to postprocess
        result['dataset_belong'] = np.concatenate([
            np.full(len(self._train_dataset), 'train'),
            np.full(len(self._valid_dataset), 'valid'),
            np.full(len(self._test_dataset), 'test')
        ])

        # denormalize pred result
        if self._train_dataset.scalers.get("y",None):
            _, result['denormalized_pred_result'] = self._train_dataset.inverse_transform_x_y(None,result)
        else:
            result['denormalized_pred_result'] = result["Pred_" + self._train_dataset.y_column[0]]

        if filename is not None:
            result.to_csv(filename, index=False)
            print(f"Result saved as {os.path.abspath(filename)}")
        return result

    def coefficient_result(self):
        """
        get the Coefficients of each argument in dataset

        Returns
        -------
        dataframe
            the Pandas dataframe of the coefficient of each argument in dataset
        """
        result_data = self.reg_result(only_return=True)
        result_data['id'] = result_data['id'].astype(np.int64)
        data = pd.concat([self._train_dataset.dataframe, self._valid_dataset.dataframe, self._test_dataset.dataframe],ignore_index=True)
        data.set_index('id', inplace=True)
        result_data.set_index('id', inplace=True)
        result_data = result_data.join(data)
        return result_data

    def __str__(self) -> str:
        r"""
        Return a string representation of the model.

        This method provides a human-readable description of the model,
        including its name and structure.

        Returns
        -------
        str
            A string containing the model name and structure

        Examples
        --------
        >>> print(model)
        Model Name: GNNWR_20230101-120000
        Model Structure: SWNN(...)
        """
        return f"Model Name: {self._model_name}\nModel Structure: {self._model}"

    def __repr__(self) -> str:
        r"""
        Return an unambiguous string representation of the model.

        This method provides a developer-friendly representation of the model
        that could ideally be used to recreate the object.
        """
        return f"{self.__class__.__name__}(model_name={self._model_name})"


class GTNNWR(GNNWR):
    """
    GTNNWR model is a model based on GNNWR and STPNN, which is a model that can be used to solve the problem of
    spatial-temporal non-stationarity.

    Parameters
    ----------
    train_dataset : BaseDataset
        the dataset for training
    valid_dataset : BaseDataset
        the dataset for validation
    test_dataset : BaseDataset
        the dataset for test
    dense_layers : list
        the dense layers of the SWNN (default: ``None``)
    start_lr : float
        the start learning rate (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"AdamW"``)
        choose from "SGD","Adam","RMSprop","AdamW","Adadelta"
    model_name : str
        the name of the model (default: ``"GTNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")``)
    model_save_path : str
        the path of the model (default: ``"../gtnnwr_models"``)
    write_path : str
        the path of the log (default: ``"../gtnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")``)
    use_gpu : bool
        whether use gpu or not (default: ``True``)
    use_ols : bool
        whether use ols or not (default: ``True``)
    log_path : str
        the path of the log (default: ``"../gtnnwr_logs"``)
    log_file_name : str
        the name of the log (default: ``"gtnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"``)
    log_level : int
        the level of the log (default: ``logging.INFO``)
    optimizer_params : dict, optional
        the params of the optimizer and the scheduler (default: ``None``)
    tensorboard_mode : bool, optional
        whether use tensorboard or not (default: ``True``)
    log_mode : bool, optional
        whether use log or not (default: ``True``)
    kwargs:dict
        the params of the model (default: ``None``)
        - drop_out : float
            the drop out rate of the model (default: ``0.2``)
        - batch_norm : bool, optional
            whether use batch normalization (default: ``True``)
        - activate_func : torch.nn
            the activate function of the model (default: ``nn.PReLU(init=0.4)``)
        - stpnn_params : dict, optional
            the params of STPNN (default: ``None``)
            - insize : int
                input size of STPNN(must be positive)
            - outsize : int
                Output size of STPNN(must be positive)
            - dense_layer : list, optional
                a list of dense layers of STPNN (default: ``None``)
            - batch_norm : bool, optional
                whether use batch normalization in STPNN or not (default: ``True``)
            - drop_out : float
                the drop out rate of STPNN (default: ``0.2``)
            - activate_func : torch.nn
                the activate function of STPNN (default: ``nn.ReLU()``)
    """

    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 dense_layers=None,
                 start_lr: float = .1,
                 optimizer="AdamW",
                 model_name="GTNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
                 model_save_path="../gtnnwr_models",
                 write_path="../gtnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 use_gpu: bool = True,
                 use_ols: bool = True,
                 log_path: str = "../gtnnwr_logs/",
                 log_file_name: str = "gtnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
                 log_level: int = logging.INFO,
                 optimizer_params=None,
                 tensorboard_mode=True,
                 log_mode=True,
                 **kwargs
                 ):

        if dense_layers is None:
            dense_layers = []

        super(GTNNWR, self).__init__(
            train_dataset,
            valid_dataset,
            test_dataset,
            dense_layers[1],
            start_lr,
            optimizer,
            model_name,
            model_save_path,
            write_path,
            use_gpu,
            use_ols,
            log_path,
            log_file_name,
            log_level,
            optimizer_params,
            tensorboard_mode,
            log_mode,
            kwargs=kwargs
        )

        self.stpnn_params = kwargs.get("stpnn_params",{
            "insize": 2,
            "outsize": 1,
        })

        if isinstance(dense_layers[0], list) and len(dense_layers[0]) > 0:
            self.stpnn_params["dense_layer"] = dense_layers[0]
            dense_layers = dense_layers[1]
            warnings.warn("Future versions will only use dense_layers to set the network structure of SWNN, the network structure of STPNN will be set by STPNN_params", FutureWarning)

        self._model = nn.Sequential(
            STPNN(
                self.stpnn_params,
            ),
            SWNN(
                dense_layers, 
                self.stpnn_params["outsize"] * self._insize, 
                self._outsize, 
                self._drop_out,
                self._activate_func, 
                self._batch_norm
            )
        )
        self.init_optimizer(self._optimizer_name, self._optimizer_params)