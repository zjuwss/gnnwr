import datetime
import math
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter  # 用于保存训练过程
from tqdm import tqdm, trange

import logging
from .networks import SWNN, STPNN, STNN_SPNN
from .utils import OLS, DIAGNOSIS


# 23.6.8_TODO: 寻找合适的优化器  考虑SGD+学习率调整  输出权重
class GNNWR:
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
            model_name="GNNWR_" + datetime.date.today().strftime("%Y%m%d-%H%M%S"),
            model_save_path="../gnnwr_models",
            write_path="../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            use_gpu: bool = True,
            use_ols: bool = True,
            log_path="../gnnwr_logs/",
            log_file_name="gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
            log_level=logging.INFO,
            optimizer_params=None,
    ):
        """
        Parameters
        ----------
        train_dataset       : baseDataset
                              dataset for tarining the network

        valid_dataset       : baseDataset
                              dataset for validation

        test_dataset        : baseDataset
                              dataset for test the trained network

        dense_layers        : list
                              neural size for each dense layer

        start_lr            : numbers
                              start learning rate

        optimizer           : string
                              type of optimizer used to change the weights or learning rates of neural network;
                              avaliable options:
                                'SGD'
                                'Adam'
                                'RMSprop'
                                'Adagrad'
                                'Adadelta'

        drop_out            : numbers
                              dropout rate at each training step

        batch_norm          : bool
                              True for use batch normalization method in training

        activate_func       : class
                              activate function defined in torch.nn

        model_name           : string
                              name of the model

        model_save_path     : string
                                path to save the trained model

        write_path          : string
                                path to save the training process

        log_path            : string
                                path to save the log file
        use_gpu             : bool
                                True for use gpu
        """
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

        self._weight = OLS(
            train_dataset.scaledDataframe, train_dataset.x, train_dataset.y).params  # OLS for weight
        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # layer to multiply weight,coefficients, and model output
        if use_ols:
            self._out.weight = nn.Parameter(torch.tensor([self._weight]).to(
                torch.float32), requires_grad=False)  # define the weight
        else:
            self._out.weight = nn.Parameter(torch.tensor(np.ones((1, self._outsize))).to(
                torch.float32), requires_grad=False)  # define the weight
        self._criterion = nn.MSELoss()  # loss function
        self._trainLossList = []  # record the loss in training process
        self._validLossList = []  # record the loss in validation process
        self._epoch = 0  # current epoch
        self._bestr2 = float('-inf')  # best r2
        self._noUpdateEpoch = 0  # number of epochs without update
        self._modelName = model_name  # model name
        self._modelSavePath = model_save_path  # model save path
        self._train_diagnosis = None  # diagnosis of training
        self._test_diagnosis = None  # diagnosis of test
        self._valid_r2 = None  # r2 of validation
        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False
        self.init_optimizer(optimizer, optimizer_params)  # initialize the optimizer

    def init_optimizer(self, optimizer, optimizer_params=None):
        # initialize the optimizer
        if optimizer == "SGD":
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=1)
        elif optimizer == "Adam":
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._start_lr)
        elif optimizer == "RMSprop":
            self._optimizer = optim.RMSprop(
                self._model.parameters(), lr=self._start_lr)
        elif optimizer == "Adagrad":
            self._optimizer = optim.Adagrad(
                self._model.parameters(), lr=self._start_lr)
        elif optimizer == "Adadelta":
            self._optimizer = optim.Adadelta(
                self._model.parameters(), lr=self._start_lr)
        else:
            raise ValueError("Invalid Optimizer")
        self._optimizer_name = optimizer  # optimizer name

        # lr scheduler
        if self._optimizer_name == "SGD":
            if optimizer_params is None:
                optimizer_params = {}
            maxlr = optimizer_params.get("maxlr", 0.1)
            minlr = optimizer_params.get("minlr", 0.01)
            upepoch = optimizer_params.get("upepoch", 100)
            uprate = (maxlr - minlr) / upepoch * 1000
            decayepoch = optimizer_params.get("decayepoch", 200)
            decayrate = optimizer_params.get("decayrate", 0.1)
            lamda_lr = lambda epoch: (epoch // 1000) * uprate + minlr if epoch < upepoch else (
                maxlr if epoch < decayepoch else maxlr * (decayrate ** (epoch - decayepoch)))
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lamda_lr)
        else:
            self._scheduler = optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=[500, 1000, 2000, 4000], gamma=0.5)

    def __train(self):
        """
        train the network
        """
        self._model.train()  # set the model to train mode
        train_loss = 0  # initialize the loss
        data_loader = self._train_dataset.dataloader  # get the data loader
        maxindex = len(data_loader)  # get the number of batches
        weight_all = torch.tensor([]).to(torch.float32)
        x_true = torch.tensor([]).to(torch.float32)
        y_true = torch.tensor([]).to(torch.float32)
        y_pred = torch.tensor([]).to(torch.float32)
        for index, (data, coef, label, id) in enumerate(data_loader):
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
            weight_all = torch.cat((weight_all, weight.mul(torch.tensor(self._weight).to(torch.float32).to(device))), 0)
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
            for data, coef, label, id in data_loader:
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
            if r2 > self._bestr2:  # if the R square is better than the best R square,record the R square and save the model
                self._bestr2 = r2
                self._noUpdateEpoch = 0
                if not os.path.exists(self._modelSavePath):
                    os.mkdir(self._modelSavePath)
                torch.save(self._model, self._modelSavePath + '/' + self._modelName + ".pkl")
            else:
                self._noUpdateEpoch += 1

    def __test(self):
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
            for data, coef, label, id in data_loader:
                device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
                data, coef, label = data.to(device), coef.to(device), label.to(device)
                x_data, y_data, y_pred, weight_all = x_data.to(device), y_data.to(device), y_pred.to(
                    device), weight_all.to(device)
                # data,label = data.view(data.shape[0],-1),label.view(data.shape[0],-1)
                x_data = torch.cat((x_data, coef), 0)
                y_data = torch.cat((y_data, label), 0)
                weight = self._model(data)
                weight_all = torch.cat(
                    (weight_all, weight.mul(torch.tensor(self._weight).to(torch.float32).to(device))), 0)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
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

    def run(self, max_epoch=1, early_stop=-1):
        """
        run the model
        """
        # 23.6.8_TODO: 丰富输出信息  输出Log文件
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

            # print("Epoch: ", epoch + 1)
            # train the network
            # record the information of the training process
            self.__train()
            # validate the network
            # record the information of the validation process
            self.__valid()
            # out put log every 50 epoch:
            if (epoch + 1) % 50 == 0:
                print("\nEpoch: ", epoch + 1)
                print("learning rate: ", self._optimizer.param_groups[0]['lr'])
                print("Train Loss: ", self._trainLossList[-1])
                print("Train R2: {:.5f}".format(self._train_diagnosis.R2().data))
                print("Train RMSE: {:.5f}".format(self._train_diagnosis.RMSE().data))
                print("Train AIC: {:.5f}".format(self._train_diagnosis.AIC()))
                print("Train AICc: {:.5f}".format(self._train_diagnosis.AICc()))
                print("Valid Loss: ", self._validLossList[-1])
                print("Valid R2: {:.5f}".format(self._valid_r2), "\n")
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
                break
        self.load_model(self._modelSavePath + '/' + self._modelName + ".pkl")
        print("Best_r2:", self._bestr2)

    def predict(self, dataset):
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
        # result = dataset.rescale(result)
        return result

    def load_model(self, path, use_dict=False):
        # load model
        if use_dict:
            data = torch.load(path).state_dict()
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path)
        self.__istrained = True

    def getLoss(self):
        """
        get network's loss
        """
        return self._trainLossList, self._validLossList

    def add_graph(self):
        """
        add graph to tensorboard
        """
        for data, coef, label, id in self._train_dataset.dataloader:
            if self._use_gpu:
                data = data.cuda()
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
                data = data.cpu()
            self._writer.add_graph(self._model, data)
            break
        print("Add Graph Successfully")

    def result(self, path=None, use_dict=False):
        """
        get the result of the model
        """
        # load model
        if path is None:
            path = self._modelSavePath + "/" + self._modelName + ".pkl"
        if use_dict:
            data = torch.load(path).state_dict()
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path)
        with torch.no_grad():
            self.__test()
        print("Test Loss: ", self.__testLoss, " Test R2: ", self.__testr2)
        logging.info("Test Loss: " + str(self.__testLoss) + "; Test R2: " + str(self.__testr2))
        # print result
        # basic information
        print("--------------------Result Table--------------------\n")
        print("Model Name:           |", self._modelName)
        print("Model Structure:      |\n", self._model)
        print("Optimizer:            |\n", self._optimizer)
        print("independent variable: |", self._train_dataset.x)
        print("dependent variable:   |", self._train_dataset.y)
        print("\n----------------------------------------------------\n")
        # OLS
        print("OLS:  |", self._weight)
        # Diagnostics
        print("R2:   |", self.__testr2)
        print("RMSE: | {:5f}".format(self._test_diagnosis.RMSE().data))
        print("AIC:  | {:5f}".format(self._test_diagnosis.AIC()))
        print("AICc: | {:5f}".format(self._test_diagnosis.AICc()))
        print("F1:   | {:5f}".format(self._test_diagnosis.F1_GNN().data))

    def reg_result(self, filename, model_path=None, use_dict=False):

        if model_path is None:
            model_path = self._modelSavePath + "/" + self._modelName + ".pkl"
        if use_dict:
            data = torch.load(model_path).state_dict()
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(model_path)
        device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
        result = torch.tensor([]).to(torch.float32).to(device)
        with torch.no_grad():
            for data, coef, label, id in self._train_dataset.dataloader:
                data, coef, label, id = data.to(device), coef.to(device), label.to(device), id.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                weight = self._model(data).mul(torch.tensor(self._weight).to(torch.float32).to(device))
                output = torch.cat((weight, output, id), dim=1)
                result = torch.cat((result, output), 0)
            for data, coef, label, id in self._valid_dataset.dataloader:
                data, coef, label, id = data.to(device), coef.to(device), label.to(device), id.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                weight = self._model(data).mul(torch.tensor(self._weight).to(torch.float32).to(device))
                output = torch.cat((weight, output, id), dim=1)
                result = torch.cat((result, output), 0)
            for data, coef, label, id in self._test_dataset.dataloader:
                data, coef, label, id = data.to(device), coef.to(device), label.to(device), id.to(device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                weight = self._model(data).mul(torch.tensor(self._weight).to(torch.float32).to(device))
                output = torch.cat((weight, output, id), dim=1)
                result = torch.cat((result, output), 0)
        result = result.cpu().detach().numpy()
        columns = self._train_dataset.x
        for i in range(len(columns)):
            columns[i] = "weight_" + columns[i]
        columns.append("bias")
        columns = columns + self._train_dataset.y + self._train_dataset.id
        result = pd.DataFrame(result, columns=columns)
        result.to_csv(filename, index=False)


class GTNNWR(GNNWR):
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
        if optimizer_params is None:
            optimizer_params = {}
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
        if train_dataset.is_need_STNN:
            self._model = nn.Sequential(STNN_SPNN(train_dataset.temporal.shape[-1], self.STNN_outsize,
                                                  train_dataset.distances.shape[-1], self.SPNN_outsize),
                                        STPNN(dense_layers[0], self.STNN_outsize + self.SPNN_outsize,
                                              self._STPNN_out, drop_out, batch_norm=False),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        else:
            self._model = nn.Sequential(STPNN(dense_layers[0], insize, self._STPNN_out, drop_out, batch_norm=False),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        self.init_optimizer(optimizer, optimizer_params)
        print(self._model)
