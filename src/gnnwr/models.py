import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter  # to save the process of the model
from tqdm import trange
from collections import OrderedDict
import logging
from .networks import SWNN, STPNN, STNN_SPNN
from .utils import OLS, DIAGNOSIS


class GNNWR:
    r"""
    GNNWR(Geographically neural network weighted regression) is a model to address spatial non-stationarity in various domains with complex geographical processes,
    which comes from the paper `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`__.

    Parameters
    ----------
    train_dataset : baseDataset
        the dataset of training
    valid_dataset : baseDataset
        the dataset of validation
    test_dataset : baseDataset
        the dataset of testing
    dense_layers : list
        the dense layers of the model (default: ``None``)

        Default structure is a geometric sequence of power of 2, the minimum is 2, and the maximum is the power of 2 closest to the number of neurons in the input layer.
        
        i.e. ``[2,4,8,16,32,64,128,256]``
    start_lr : float
        the start learning rate of the model (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"Adagrad"``)
        choose from "SGD","Adam","RMSprop","Adagrad","Adadelta"
    drop_out : float
        the drop out rate of the model (default: ``0.2``)
    batch_norm : bool, optional
        whether use batch normalization (default: ``True``)
    activate_func : torch.nn
        the activate function of the model (default: ``nn.PReLU(init=0.4)``)
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
        the params of the optimizer and the scheduler (default: ``None``)

        if optimizer is SGD, the params are:

            | maxlr: float, the max learning rate (default: ``0.1``)

            | minlr: float, the min learning rate (default: ``0.01``)

            | upepoch: int, the epoch of learning rate up (default: ``10000``)

            | decayepoch: int, the epoch of learning rate decay (default: ``20000``)

            | decayrate: float, the rate of learning rate decay (default: ``0.1``)

            | stop_change_epoch: int, the epoch of learning rate stop change (default: ``30000``)

            | stop_lr: float, the learning rate when stop change (default: ``0.001``)

        if optimizer is Other, the params are:

            | scheduler: str, the name of the scheduler (default: ``"CosineAnnealingWarmRestarts"``) in {``"MultiStepLR","CosineAnnealingLR","CosineAnnealingWarmRestarts"``}

            | scheduler_milestones: list, the milestones of the scheduler MultiStepLR (default: ``[500,1000,2000,4000]``)

            | scheduler_gamma: float, the gamma of the scheduler MultiStepLR (default: ``0.5``)

            | scheduler_T_max: int, the T_max of the scheduler CosineAnnealingLR (default: ``1000``)

            | scheduler_eta_min: float, the eta_min of the scheduler CosineAnnealingLR and CosineAnnealingWarmRestarts (default: ``0.01``)

            | scheduler_T_0: int, the T_0 of the scheduler CosineAnnealingWarmRestarts (default: ``100``)

            | scheduler_T_mult: int, the T_mult of the scheduler CosineAnnealingWarmRestarts (default: ``3``)


    """

    def __init__(
            self,
            train_dataset,
            valid_dataset,
            test_dataset,
            dense_layers=None,
            start_lr: float = .1,
            optimizer="Adagrad",
            drop_out=0.2,
            batch_norm=True,
            activate_func=nn.PReLU(init=0.4),
            model_name="GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
            model_save_path="../gnnwr_models",
            write_path="../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            use_gpu: bool = True,
            use_ols: bool = True,
            log_path="../gnnwr_logs/",
            log_file_name="gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
            log_level=logging.INFO,
            optimizer_params=None
    ):
        self._train_dataset = train_dataset  # train dataset
        self._valid_dataset = valid_dataset  # valid dataset
        self._test_dataset = test_dataset  # test dataset
        self._dense_layers = dense_layers  # structure of layers
        self._start_lr = start_lr  # initial learning rate
        self._insize = train_dataset.datasize  # size of input layer
        self._outsize = train_dataset.coefsize  # size of output layer
        self._writer = SummaryWriter(write_path)  # summary writer
        self._drop_out = drop_out  # drop_out ratio
        self._batch_norm = batch_norm  # batch normalization
        self._activate_func = activate_func  # activate function , default: PRelu(0.4)
        self._model = SWNN(self._dense_layers, self._insize, self._outsize,
                           self._drop_out, self._activate_func, self._batch_norm)  # model
        self._log_path = log_path  # log path
        self._log_file_name = log_file_name  # log file
        self._log_level = log_level  # log level
        self.__istrained = False  # whether the model is trained

        self._coefficient = OLS(
            train_dataset.scaledDataframe, train_dataset.x, train_dataset.y).params  # coefficients of OLS
        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # layer to multiply OLS coefficients and model output
        if use_ols:
            self._out.weight = nn.Parameter(torch.tensor([self._coefficient]).to(
                torch.float32), requires_grad=False)  # define the weight
        else:
            self._coefficient = np.ones((1, self._outsize))
            self._out.weight = nn.Parameter(torch.tensor(np.ones((1, self._outsize))).to(
                torch.float32), requires_grad=False)  # define the weight
        self._criterion = nn.MSELoss()  # loss function
        self._trainLossList = []  # record the loss in training process
        self._validLossList = []  # record the loss in validation process
        self._epoch = 0  # current epoch
        self._bestr2 = float('-inf')  # best r2
        self._besttrainr2 = float('-inf')  # best train r2
        self._noUpdateEpoch = 0  # number of epochs without update
        self._modelName = model_name  # model name
        self._modelSavePath = model_save_path  # model save path
        self._train_diagnosis = None  # diagnosis of training
        self._test_diagnosis = None  # diagnosis of test
        self._valid_r2 = None  # r2 of validation
        self.result_data = None
        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False
        self._optimizer = None
        self._scheduler = None
        self._optimizer_name = None
        self.init_optimizer(optimizer, optimizer_params)  # initialize the optimizer

    def init_optimizer(self, optimizer, optimizer_params=None):
        r"""
        initialize the optimizer

        Parameters
        ----------
        optimizer : str
            the optimizer of the model (default: ``"Adagrad"``)
            choose from "SGD","Adam","RMSprop","Adagrad","Adadelta"
        optimizer_params : dict, optional
            the params of the optimizer and the scheduler (default: ``None``)

            if optimizer is SGD, the params are:

                | maxlr: float, the max learning rate (default: ``0.1``)

                | minlr: float, the min learning rate (default: ``0.01``)

                | upepoch: int, the epoch of learning rate up (default: ``10000``)

                | decayepoch: int, the epoch of learning rate decay (default: ``20000``)

                | decayrate: float, the rate of learning rate decay (default: ``0.1``)

                | stop_change_epoch: int, the epoch of learning rate stop change (default: ``30000``)

                | stop_lr: float, the learning rate when stop change (default: ``0.001``)

            if optimizer is Other, the params are:

                | scheduler: str, the name of the scheduler (default: ``"CosineAnnealingWarmRestarts"``) in {``"MultiStepLR","CosineAnnealingLR","CosineAnnealingWarmRestarts"``}

                | scheduler_milestones: list, the milestones of the scheduler MultiStepLR (default: ``[500,1000,2000,4000]``)

                | scheduler_gamma: float, the gamma of the scheduler MultiStepLR (default: ``0.5``)

                | scheduler_T_max: int, the T_max of the scheduler CosineAnnealingLR (default: ``1000``)

                | scheduler_eta_min: float, the eta_min of the scheduler CosineAnnealingLR and CosineAnnealingWarmRestarts (default: ``0.01``)

                | scheduler_T_0: int, the T_0 of the scheduler CosineAnnealingWarmRestarts (default: ``100``)

                | scheduler_T_mult: int, the T_mult of the scheduler CosineAnnealingWarmRestarts (default: ``3``)
        """
        # initialize the optimizer
        if optimizer == "SGD":
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=1, weight_decay=1e-3)
        elif optimizer == "Adam":
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._start_lr, weight_decay=1e-3)
        elif optimizer == "RMSprop":
            self._optimizer = optim.RMSprop(
                self._model.parameters(), lr=self._start_lr)
        elif optimizer == "Adagrad":
            self._optimizer = optim.Adagrad(
                self._model.parameters(), lr=self._start_lr)
        elif optimizer == "Adadelta":
            self._optimizer = optim.Adadelta(
                self._model.parameters(), lr=self._start_lr, weight_decay=1e-3)
        else:
            raise ValueError("Invalid Optimizer")
        self._optimizer_name = optimizer  # optimizer name

        # lr scheduler
        if self._optimizer_name == "SGD":
            if optimizer_params is None:
                optimizer_params = {}
            maxlr = optimizer_params.get("maxlr", 0.1)
            minlr = optimizer_params.get("minlr", 0.01)
            upepoch = optimizer_params.get("upepoch", 10000)
            uprate = (maxlr - minlr) / upepoch * (upepoch // 10)
            decayepoch = optimizer_params.get("decayepoch", 20000)
            decayrate = optimizer_params.get("decayrate", 0.95)
            stop_change_epoch = optimizer_params.get("stop_change_epoch", 30000)
            stop_lr = optimizer_params.get("stop_lr", 0.001)
            lamda_lr = lambda epoch: (epoch // (upepoch // 10)) * uprate + minlr if epoch < upepoch else (
                maxlr if epoch < decayepoch else maxlr * (decayrate ** ((epoch - decayepoch)//10))) if epoch < stop_change_epoch else stop_lr
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lamda_lr)
        else:
            if optimizer_params is None:
                optimizer_params = {}
            scheduler = optimizer_params.get("scheduler", "CosineAnnealingWarmRestarts")
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
            else:
                raise ValueError("Invalid Scheduler")

    def __train(self):
        """
        train the network
        """
        self._model.train()  # set the model to train mode
        train_loss = 0  # initialize the loss
        data_loader = self._train_dataset.dataloader  # get the data loader
        weight_all = torch.tensor([]).to(torch.float32)
        x_true = torch.tensor([]).to(torch.float32)
        y_true = torch.tensor([]).to(torch.float32)
        y_pred = torch.tensor([]).to(torch.float32)
        for index, (data, coef, label, data_index) in enumerate(data_loader):
            # move the data to gpu
            device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
            data, coef, label = data.to(device), coef.to(device), label.to(device)
            weight_all, x_true, y_true, y_pred = weight_all.to(device), x_true.to(device), y_true.to(device), y_pred.to(
                device)

            self._optimizer.zero_grad()  # zero the gradient
            if self._optimizer_name == "Adagrad":
                # move optimizer state to gpu
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

            x_true = torch.cat((x_true, coef), 0)
            y_true = torch.cat((y_true, label), 0)
            weight = self._model(data)
            weight_all = torch.cat((weight_all, weight.to(torch.float32)), 0)
            output = self._out(weight.mul(coef.to(torch.float32)))
            y_pred = torch.cat((y_pred, output), 0)
            loss = self._criterion(output, label)  # calculate the loss
            loss.backward()  # back propagation
            self._optimizer.step()  # update the parameters
            if isinstance(data, list):
                train_loss += loss.item() * data[0].size(0)
            else:
                train_loss += loss.item() * data.size(0)  # accumulate the loss

        self._train_diagnosis = DIAGNOSIS(weight_all, x_true, y_true, y_pred)
        train_loss /= self._train_dataset.datasize  # calculate the average loss
        self._trainLossList.append(train_loss)  # record the loss

    def __valid(self):
        """
        validate the network
        """
        self._model.eval()  # set the model to validation mode
        val_loss = 0  # initialize the loss
        label_list = np.array([])  # label list
        out_list = np.array([])  # output list
        data_loader = self._valid_dataset.dataloader  # get the data loader

        with torch.no_grad():  # disable gradient calculation
            for data, coef, label, data_index in data_loader:
                device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
                data, coef, label = data.to(device), coef.to(device), label.to(device)
                # weight = self._model(data)
                output = self._out(self._model(
                    data).mul(coef.to(torch.float32)))
                loss = self._criterion(output, label)  # calculate the loss
                out_list = np.append(
                    out_list, output.view(-1).cpu().detach().numpy())  # add the output to the list
                label_list = np.append(
                    label_list, label.view(-1).cpu().numpy())  # add the label to the list
                if isinstance(data, list):
                    val_loss += loss.item() * data[0].size(0)
                else:
                    val_loss += loss.item() * data.size(0)  # accumulate the loss
            val_loss /= len(self._valid_dataset)  # calculate the average loss
            self._validLossList.append(val_loss)  # record the loss
            try:
                r2 = r2_score(label_list, out_list)  # calculate the R square
            except:
                print(label_list)
                print(out_list)
            self._valid_r2 = r2
            if r2 > self._bestr2:
                # if the R square is better than the best R square,record the R square and save the model
                self._bestr2 = r2
                self._besttrainr2 = self._train_diagnosis.R2().data
                self._noUpdateEpoch = 0
                if not os.path.exists(self._modelSavePath):
                    os.mkdir(self._modelSavePath)
                torch.save(self._model, self._modelSavePath + '/' + self._modelName + ".pkl")
            else:
                self._noUpdateEpoch += 1

    def __test(self):
        """
        test the network
        """
        self._model.eval()
        test_loss = 0
        label_list = np.array([])
        out_list = np.array([])
        data_loader = self._test_dataset.dataloader
        x_data = torch.tensor([]).to(torch.float32)
        y_data = torch.tensor([]).to(torch.float32)
        y_pred = torch.tensor([]).to(torch.float32)
        weight_all = torch.tensor([]).to(torch.float32)
        with torch.no_grad():
            for data, coef, label, data_index in data_loader:
                device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
                data, coef, label = data.to(device), coef.to(device), label.to(device)
                x_data, y_data, y_pred, weight_all = x_data.to(device), y_data.to(device), y_pred.to(
                    device), weight_all.to(device)
                # data,label = data.view(data.shape[0],-1),label.view(data.shape[0],-1)
                x_data = torch.cat((x_data, coef), 0)
                y_data = torch.cat((y_data, label), 0)
                weight = self._model(data)
                weight_all = torch.cat(
                    (weight_all, weight.to(torch.float32).to(device)), 0)
                output = self._out(weight.mul(coef.to(torch.float32)))
                y_pred = torch.cat((y_pred, output), 0)
                loss = self._criterion(output, label)

                out_list = np.append(
                    out_list, output.view(-1).cpu().detach().numpy())  # add the output to the list
                label_list = np.append(
                    label_list, label.view(-1).cpu().numpy())  # add the label to the list
                if isinstance(data, list):
                    test_loss += loss.item() * data[0].size(0)
                else:
                    test_loss += loss.item() * data.size(0)  # accumulate the loss
            test_loss /= len(self._test_dataset)
            self.__testLoss = test_loss
            self.__testr2 = r2_score(label_list, out_list)
            self._test_diagnosis = DIAGNOSIS(weight_all, x_data, y_data, y_pred)

    def run(self, max_epoch=1, early_stop=-1, print_frequency=50, show_detailed_info=True):
        """
        train the model and validate the model

        Parameters
        ----------
        max_epoch : int
            the max epoch of the training (default: ``1``)
        early_stop : int
            if the model has not been updated for ``early_stop`` epochs, the training will stop (default: ``-1``)

            if ``early_stop`` is ``-1``, the training will not stop until the max epoch
        print_frequency : int
            the frequency of printing the information (default: ``50``)

        show_detailed_info : bool
            if ``True``, the detailed information will be shown (default: ``True``)
        """
        self.__istrained = True
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        # create file
        if not os.path.exists(self._log_path):
            os.mkdir(self._log_path)
        file_str = self._log_path + self._log_file_name
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            filename=file_str, level=logging.INFO)
        for epoch in trange(0, max_epoch):
            self._epoch = epoch
            # train the network
            # record the information of the training process
            self.__train()
            # validate the network
            # record the information of the validation process
            self.__valid()
            # out put log every {print_frequency} epoch:
            if (epoch + 1) % print_frequency == 0:
                if show_detailed_info:
                    print("\nEpoch: ", epoch + 1)
                    print("learning rate: ", self._optimizer.param_groups[0]['lr'])
                    print("Train Loss: ", self._trainLossList[-1])
                    print("Train R2: {:.5f}".format(self._train_diagnosis.R2().data))
                    print("Train RMSE: {:.5f}".format(self._train_diagnosis.RMSE().data))
                    print("Train AIC: {:.5f}".format(self._train_diagnosis.AIC()))
                    print("Train AICc: {:.5f}".format(self._train_diagnosis.AICc()))
                    print("Valid Loss: ", self._validLossList[-1])
                    print("Valid R2: {:.5f}".format(self._valid_r2), "\n")
                    print("Best R2: {:.5f}".format(self._bestr2), "\n")
                else:
                    print("\nEpoch: ", epoch + 1)
                    print(
                        "Train R2: {:.5f}  Valid R2: {:.5f}  Best R2: {:.5f}\n".format(self._train_diagnosis.R2().data,
                                                                                       self._valid_r2, self._bestr2))
            self._scheduler.step()  # update the learning rate
            # tensorboard
            self._writer.add_scalar('Training/Learning Rate', self._optimizer.param_groups[0]['lr'], self._epoch)
            self._writer.add_scalar('Training/Loss', self._trainLossList[-1], self._epoch)
            self._writer.add_scalar('Training/R2', self._train_diagnosis.R2().data, self._epoch)
            self._writer.add_scalar('Training/RMSE', self._train_diagnosis.RMSE().data, self._epoch)
            self._writer.add_scalar('Training/AIC', self._train_diagnosis.AIC(), self._epoch)
            self._writer.add_scalar('Training/AICc', self._train_diagnosis.AICc(), self._epoch)
            self._writer.add_scalar('Validation/Loss', self._validLossList[-1], self._epoch)
            self._writer.add_scalar('Validation/R2', self._valid_r2, self._epoch)
            self._writer.add_scalar('Validation/Best R2', self._bestr2, self._epoch)

            # log output
            log_str = "Epoch: " + str(epoch + 1) + \
                      "; Train Loss: " + str(self._trainLossList[-1]) + \
                      "; Train R2: {:5f}".format(self._train_diagnosis.R2().data) + \
                      "; Train RMSE: {:5f}".format(self._train_diagnosis.RMSE().data) + \
                      "; Train AIC: {:5f}".format(self._train_diagnosis.AIC()) + \
                      "; Train AICc: {:5f}".format(self._train_diagnosis.AICc()) + \
                      "; Valid Loss: " + str(self._validLossList[-1]) + \
                      "; Valid R2: " + str(self._valid_r2) + \
                      "; Learning Rate: " + str(self._optimizer.param_groups[0]['lr'])
            logging.info(log_str)
            if 0 < early_stop < self._noUpdateEpoch:  # stop when the model has not been updated for long time
                print("Training stop! Model has not been improved for over {} epochs.".format(early_stop))
                break
        self.load_model(self._modelSavePath + '/' + self._modelName + ".pkl")
        self.result_data = self.getCoefs()
        print("Best_r2:", self._bestr2)

    def predict(self, dataset):
        """
        predict the result of the dataset

        Parameters
        ----------
        dataset : baseDataset,predictDataset
            the dataset to be predicted
        
        Returns
        -------
        dataframe
            the Pandas dataframe of the dataset with the predicted result
        """
        data_loader = dataset.dataloader
        if not self.__istrained:
            print("WARNING! The model hasn't been trained or loaded!")
        self._model.eval()
        result = np.array([])
        with torch.no_grad():
            for data, coef in data_loader:
                if self._use_gpu:
                    data, coef = data.cuda(), coef.cuda()
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                output = output.view(-1).cpu().detach().numpy()
                result = np.append(result, output)
        dataset.dataframe['pred_result'] = result
        dataset.pred_result = result
        return dataset.dataframe

    def predict_coef(self, dataset):
        """
        predict the spatial coefficient of the independent variable

        Parameters
        ----------
        dataset : baseDataset,predictDataset
            the dataset to be predicted

        Returns
        -------
        dataframe
            the Pandas dataframe of the dataset with the predicted spatial coefficient
        """
        data_loader = dataset.dataloader
        if not self.__istrained:
            print("WARNING! The model hasn't been trained or loaded!")
        self._model.eval()
        result = torch.tensor([]).to(torch.float32)
        with torch.no_grad():
            for data, coef in data_loader:
                if self._use_gpu:
                    result, data, coef = result.cuda(), data.cuda(), coef.cuda()
                    ols_w = torch.tensor(self._coefficient).to(torch.float32).cuda()
                else:
                    ols_w = torch.tensor(self._coefficient).to(torch.float32)
                coefficient = self._model(data).mul(ols_w)
                result = torch.cat((result, coefficient), 0)
        result = result.cpu().detach().numpy()
        return result

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
            data = torch.load(path, map_location=map_location)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location)
        if self._use_gpu:
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        self._modelSavePath = os.path.dirname(path)
        self._modelName = os.path.basename(path).split('/')[-1].split('.')[0]
        self.__istrained = True
        self.result_data = self.getCoefs()


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
            data = torch.load(path, map_location='cpu').state_dict()
        else:
            data = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        torch.save(new_state_dict, save_path)

    def getLoss(self):
        """
        get network's loss

        Returns
        -------
        list
            the list of the loss in training process and validation process
        """
        return self._trainLossList, self._validLossList

    def add_graph(self):
        """
        add the graph of the model to tensorboard
        """
        for data, coef, label, data_index in self._train_dataset.dataloader:
            if self._use_gpu:
                data = data.cuda()
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
                data = data.cpu()
            self._writer.add_graph(self._model, data)
            break
        print("Add Graph Successfully")

    def result(self, path=None, use_dict=False, map_location=None):
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
        if not self.__istrained:
            raise Exception("The model hasn't been trained or loaded!")
        if path is None:
            path = self._modelSavePath + "/" + self._modelName + ".pkl"
        if use_dict:
            data = torch.load(path, map_location=map_location)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location)
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        with torch.no_grad():
            self.__test()

        logging.info("Test Loss: " + str(self.__testLoss) + "; Test R2: " + str(self.__testr2))
        # print result
        # basic information
        print("--------------------Model Information-----------------")
        print("Model Name:           |", self._modelName)
        print("independent variable: |", self._train_dataset.x)
        print("dependent variable:   |", self._train_dataset.y)
        # OLS
        print("\nOLS coefficients: ")
        for i in range(len(self._coefficient)):
            if i == len(self._coefficient) - 1:
                print("Intercept: {:.5f}".format(self._coefficient[i]))
            else:
                print("x{}: {:.5f}".format(i, self._coefficient[i]))
        print("\n--------------------Result Information----------------")
        print("Test Loss: | {:>25.5f}".format(self.__testLoss))
        print("Test R2  : | {:>25.5f}".format(self.__testr2))
        if self._besttrainr2 is not None and self._besttrainr2 != float('-inf'):
            print("Train R2 : | {:>25.5f}".format(self._besttrainr2))
            print("Valid R2 : | {:>25.5f}".format(self._bestr2))
        print("RMSE: | {:>30.5f}".format(self._test_diagnosis.RMSE().data))
        print("AIC:  | {:>30.5f}".format(self._test_diagnosis.AIC()))
        print("AICc: | {:>30.5f}".format(self._test_diagnosis.AICc()))
        print("F1:   | {:>30.5f}".format(self._test_diagnosis.F1_Global().data))
        print("F2:   | {:>30.5f}".format(self._test_diagnosis.F2_Global().flatten()[0].data))
        F3_Local_dict = self._test_diagnosis.F3_Local()[0]
        for key in F3_Local_dict:
            width = 30-(len(key) - 4)
            print("{}: | {:>{width}.5f}".format(key, F3_Local_dict[key].data, width=width))

    def reg_result(self, filename=None, model_path=None, use_dict=False, only_return=False, map_location=None):
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
            model_path = self._modelSavePath + "/" + self._modelName + ".pkl"
        if use_dict:
            data = torch.load(model_path, map_location=map_location)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(model_path, map_location=map_location)

        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
        result = torch.tensor([]).to(torch.float32).to(device)
        with torch.no_grad():
            for data, coef, label, data_index in self._train_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            for data, coef, label, data_index in self._valid_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            for data, coef, label, data_index in self._test_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
        result = result.cpu().detach().numpy()
        columns = list(self._train_dataset.x)
        for i in range(len(columns)):
            columns[i] = "coef_" + columns[i]
        columns.append("bias")
        columns = columns + ["Pred_" + self._train_dataset.y[0]] + self._train_dataset.id
        result = pd.DataFrame(result, columns=columns)
        result[self._train_dataset.id] = result[self._train_dataset.id].astype(np.int32)
        result["Pred_" + self._train_dataset.y[0]] = result["Pred_" + self._train_dataset.y[0]].astype(np.float32)
        if only_return:
            return result
        if filename is not None:
            result.to_csv(filename, index=False)
        else:
            warnings.warn(
                "Warning! The input write file path is not set. Result is returned by function but not saved as file.",
                RuntimeWarning)
        return result

    def getCoefs(self):
        """
        get the Coefficients of each argument in dataset

        Returns
        -------
        dataframe
            the Pandas dataframe of the coefficient of each argument in dataset
        """
        result_data = self.reg_result(only_return=True)
        result_data['id'] = result_data['id'].astype(np.int64)
        data = pd.concat([self._train_dataset.dataframe, self._valid_dataset.dataframe, self._test_dataset.dataframe])
        data.set_index('id', inplace=True)
        result_data.set_index('id', inplace=True)
        result_data = result_data.join(data)
        return result_data
    def __str__(self) -> str:
        print("Model Name: ", self._modelName)
        print("Model Structure: ", self._model)
        return ""
    def __repr__(self) -> str:
        print("Model Name: ", self._modelName)
        print("Model Structure: ", self._model)
        return ""


class GTNNWR(GNNWR):
    """
    GTNNWR model is a model based on GNNWR and STPNN, which is a model that can be used to solve the problem of
    spatial-temporal non-stationarity.

    Parameters
    ----------
    train_dataset : baseDataset
        the dataset for training
    valid_dataset : baseDataset
        the dataset for validation
    test_dataset : baseDataset
        the dataset for test
    dense_layers : list
        the dense layers of the model (default: ``None``)
        | i.e. ``[[3],[128,64,32]]`` the first list in input is hidden layers of STPNN, the second one is hidden layers of SWNN.
    start_lr : float
        the start learning rate (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"Adagrad"``)
        choose from "SGD","Adam","RMSprop","Adagrad","Adadelta"
    drop_out : float
        the drop out rate of the model (default: ``0.2``)
    batch_norm : bool, optional
        whether use batch normalization (default: ``True``)
    activate_func : torch.nn
        the activate function of the model (default: ``nn.PReLU(init=0.4)``)
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
        the params of the optimizer and the scheduler (default: ``None``)

        if optimizer is SGD, the params are:

            | maxlr: float, the max learning rate (default: ``0.1``)

            | minlr: float, the min learning rate (default: ``0.01``)

            | upepoch: int, the epoch of learning rate up (default: ``10000``)

            | decayepoch: int, the epoch of learning rate decay (default: ``20000``)

            | decayrate: float, the rate of learning rate decay (default: ``0.1``)

            | stop_change_epoch: int, the epoch of learning rate stop change (default: ``30000``)

            | stop_lr: float, the learning rate when stop change (default: ``0.001``)

        if optimizer is Other, the params are:

            | scheduler: str, the name of the scheduler (default: ``"CosineAnnealingWarmRestarts"``) in {``"MultiStepLR","CosineAnnealingLR","CosineAnnealingWarmRestarts"``}

            | scheduler_milestones: list, the milestones of the scheduler MultiStepLR (default: ``[500,1000,2000,4000]``)

            | scheduler_gamma: float, the gamma of the scheduler MultiStepLR (default: ``0.5``)

            | scheduler_T_max: int, the T_max of the scheduler CosineAnnealingLR (default: ``1000``)

            | scheduler_eta_min: float, the eta_min of the scheduler CosineAnnealingLR and CosineAnnealingWarmRestarts (default: ``0.01``)

            | scheduler_T_0: int, the T_0 of the scheduler CosineAnnealingWarmRestarts (default: ``100``)

            | scheduler_T_mult: int, the T_mult of the scheduler CosineAnnealingWarmRestarts (default: ``3``)
    STPNN_outsize:int
        the output size of STPNN(default:``1``)
    STNN_SPNN_params:dict
        the params of STNN and SPNN(default:``None``)
        
        STPNN_batch_norm:bool
            
            whether use batchnorm in STNN and SPNN or not (Default:``True``)
    """

    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 dense_layers=None,
                 start_lr: float = .1,
                 optimizer="Adam",
                 drop_out=0.2,
                 batch_norm=True,
                 activate_func=nn.PReLU(init=0.4),
                 model_name="GTNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
                 model_save_path="../gtnnwr_models",
                 write_path="../gtnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 use_gpu: bool = True,
                 use_ols: bool = True,
                 log_path: str = "../gtnnwr_logs/",
                 log_file_name: str = "gtnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
                 log_level: int = logging.INFO,
                 optimizer_params=None,
                 STPNN_outsize=1,
                 STNN_SPNN_params=None,
                 ):

        if dense_layers is None:
            dense_layers = [[], []]
        super(GTNNWR, self).__init__(train_dataset, valid_dataset, test_dataset, dense_layers[1], start_lr, optimizer,
                                     drop_out, batch_norm, activate_func, model_name, model_save_path, write_path,
                                     use_gpu, use_ols, log_path, log_file_name, log_level, optimizer_params)
        self._STPNN_out = STPNN_outsize
        self._modelName = model_name  # model name
        if train_dataset.simple_distance:
            insize = 2
        else:
            insize = train_dataset.distances.shape[-1]
        if STNN_SPNN_params is None:
            STNN_SPNN_params = dict()
        self.STNN_outsize = STNN_SPNN_params.get("STNN_outsize", 1)
        self.SPNN_outsize = STNN_SPNN_params.get("SPNN_outsize", 1)
        self.STPNN_batch_norm = STNN_SPNN_params.get("STPNN_batch_norm", True)
        if train_dataset.is_need_STNN:
            self._model = nn.Sequential(STNN_SPNN(train_dataset.temporal.shape[-1], self.STNN_outsize,
                                                  train_dataset.distances.shape[-1], self.SPNN_outsize),
                                        STPNN(dense_layers[0], self.STNN_outsize + self.SPNN_outsize,
                                              self._STPNN_out, drop_out, batch_norm=self.STPNN_batch_norm),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        else:
            self._model = nn.Sequential(STPNN(dense_layers[0], insize, self._STPNN_out, drop_out,
                                              batch_norm=self.STPNN_batch_norm),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        self.init_optimizer(optimizer, optimizer_params)
