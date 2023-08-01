import datetime
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter  # 用于保存训练过程

import logging
from .networks import SWNN, STPNN
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
            model_name=datetime.date.today().strftime("%Y%m%d-%H%M%S"),
            model_save_path="../gnnwr_models",
            write_path="../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            use_gpu: bool = True,
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
        self._log_file_name = log_file_name  # log file name
        self._log_level = log_level  # log level
        self.__istrained = False # whether the model is trained

        # initialize the optimizer
        if optimizer == "SGD":
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=self._start_lr)
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
            uprate = (maxlr - minlr) / upepoch
            decayepoch = optimizer_params.get("decayepoch", 200)
            decayrate = optimizer_params.get("decayrate", 0.1)
            lamda_lr = lambda epoch: epoch * uprate + minlr if epoch < upepoch else (
                maxlr if epoch < decayepoch else maxlr * (decayrate ** (epoch - decayepoch)))
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lamda_lr)
        else:
            self._scheduler = optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=[100, 200], gamma=0.1)
        self._weight = OLS(
            train_dataset.dataframe, train_dataset.x, train_dataset.y).params  # OLS for weight
        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # layer to multiply weight,coefficients, and model output
        self._out.weight = nn.Parameter(torch.tensor([self._weight]).to(
            torch.float32), requires_grad=False)  # define the weight
        self._criterion = nn.MSELoss()  # loss function
        self._trainLossList = []  # record the loss in training process
        self._validLossList = []  # record the loss in validation process
        self._epoch = 0  # current epoch
        self._bestr2 = float('-inf')  # best r2
        self._noUpdateEpoch = 0  # number of epochs without update
        self._modelName = "GNNWR_" + model_name  # model name
        self._modelSavePath = model_save_path  # model save path
        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False

    def __train(self):
        """
        train the network
        """
        self._model.train()  # set the model to train mode
        train_loss = 0  # initialize the loss
        data_loader = self._train_dataset.dataloader  # get the data loader
        maxindex = len(data_loader)  # get the number of batches
        for index, (data, coef, label) in enumerate(data_loader):
            if self._use_gpu:
                data, coef, label = data.cuda(), coef.cuda(), label.cuda()
            # data, label = data.view(
            #     data.shape[0], -1), label.view(data.shape[0], -1)  # reshape the data
            self._optimizer.zero_grad()  # zero the gradient

            output = self._out(self._model(data).mul(coef.to(torch.float32)))
            loss = self._criterion(output, label)  # calculate the loss
            loss.backward()  # back propagation
            self._optimizer.step()  # update the parameters
            if isinstance(data, list):
                train_loss += loss.item() * data[0].size(0)
            else:
                train_loss += loss.item() * data.size(0)  # accumulate the loss

            # print the progress bar
            i = index + 1
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-50s] %d%%" % ('#' * int(i * 50.0 / maxindex), int(100.0 * i / maxindex)))
            sys.stdout.flush()

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
            for data, coef, label in data_loader:
                if self._use_gpu:
                    data, coef, label = data.cuda(), coef.cuda(), label.cuda()
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
            r2 = r2_score(label_list, out_list)  # calculate the R square
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

        with torch.no_grad():
            for data, coef, label in data_loader:
                if self._use_gpu:
                    data, coef, label = data.cuda(), coef.cuda(), label.cuda()
                # data,label = data.view(data.shape[0],-1),label.view(data.shape[0],-1)

                output = self._out(self._model(data).mul(coef.to(torch.float32)))
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
        for epoch in range(0, max_epoch):
            self._epoch = epoch
            print("Epoch: ", epoch + 1)
            self.__train()
            print("\nTrain Loss: ", self._trainLossList[-1])
            self.__valid()
            print("Valid Loss: ", self._validLossList[-1], "\n")
            # TODO tensorboard
            group = self._optimizer.param_groups[0]
            p = group['params'][0]
            if self._optimizer_name == 'Adam':
                beta1, _ = group['betas']
                state = self._optimizer.state[p]
                bias_correction1 = 1 - beta1 ** state['step']
                current_lr = group['lr'] / bias_correction1
            elif self._optimizer_name == 'SGD':
                current_lr = group['lr']
            elif self._optimizer_name == 'RMSprop':
                alpha = group['alpha']
                state = self._optimizer.state[p]
                avg_sq = state['square_avg']
                current_lr = group['lr'] / ((avg_sq.sqrt() / math.sqrt(1 - alpha ** state['step'])) + group['eps'])
            elif self._optimizer_name == 'Adadelta':
                state = self._optimizer.state[p]
                acc_delta = state['acc_delta']
                current_lr = group['lr'] * (state['square_avg'] + group['eps']).sqrt() / (
                        acc_delta + group['eps']).sqrt()
            elif self._optimizer_name == 'Adagrad':
                state = self._optimizer.state[p]
                sum_sq = state['sum']
                current_lr = group['lr'] / (sum_sq.sqrt() + group['eps'])
            else:
                raise NotImplementedError
            self._scheduler.step()

            self._writer.add_scalar('Training/Loss', self._trainLossList[-1], self._epoch)
            self._writer.add_scalar('Training/Learning Rate', current_lr, self._epoch)
            self._writer.add_scalar('Validation/Loss', self._validLossList[-1], self._epoch)
            self._writer.add_scalar('Validation/R2', self._bestr2, self._epoch)

            # log output
            log_str = "Epoch: " + str(epoch + 1) + " Train Loss: " + str(
                self._trainLossList[-1]) + " R2: " + str(self._bestr2) + " Valid Loss: " + str(
                self._validLossList[-1]) + " Learning Rate: " + str(current_lr)
            logging.info(log_str)
            if 0 < early_stop < self._noUpdateEpoch:  # stop when the model has not been updated for long time
                break
        self.__test()
        print("Test Loss: ", self.__testLoss, " Test R2: ", self.__testr2)
        logging.info("Test Loss: " + str(self.__testLoss) + " Test R2: " + str(self.__testr2))

    def predict(self, dataset):
            # 23.6.8_TODO:load_model
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
        result = dataset.rescale(result)
        return result

    def load_model(self, path, use_dict=False):
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

        :return:
        """
        for data,coef,label in self._train_dataset.dataloader:
            self._writer.add_graph(self._model,data)
            break
        print("Add Graph Successfully")


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
                 model_name=datetime.date.today().strftime("%y%m%d"),
                 model_save_path="../gtnnwr_models",
                 write_path="../gtnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 use_gpu: bool = True,
                 log_path: str = "../gtnnwr_logs/",
                 log_file_name: str = "gtnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
                 log_level: int = logging.INFO,
                 optimizer_params=None,
                 ):
        if optimizer_params is None:
            optimizer_params = {}
        if dense_layers is None:
            dense_layers = [[], []]
        super(GTNNWR, self).__init__(train_dataset, valid_dataset, test_dataset, dense_layers[1], start_lr, optimizer,
                                     drop_out, batch_norm, activate_func, model_name, model_save_path, write_path,
                                     use_gpu, log_path, log_file_name, log_level, optimizer_params)
        self._STPNN_out = 1
        self._modelName = "GTNNWR_" + model_name  # model name
        self._model = nn.Sequential(STPNN(dense_layers[0], 2, self._STPNN_out, drop_out, activate_func, batch_norm),
                                    SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                         activate_func, batch_norm))
        print(self._model)
