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
            model_name=datetime.date.today().strftime("%y%m%d"),
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
        self._insize = train_dataset.datasize  # 输入层大小，需要在dataset类中提供获取datasize的方法
        self._outsize = train_dataset.coefsize  # 输出层大小，需要在dataset类中提供获取coefsize的方法
        self._writer = SummaryWriter(write_path)  # 用于保存训练过程
        self._drop_out = drop_out  # drop_out比例
        self._batch_norm = batch_norm  # 是否进行批归一化
        self._activate_func = activate_func  # 激活函数，由用户在外定义后传入，默认为PRelu(0.4)
        self._model = SWNN(self._dense_layers, self._insize, self._outsize,
                           self._drop_out, self._activate_func, self._batch_norm)  # 网络模型
        self._log_path = log_path  # 日志文件路径
        self._log_file_name = log_file_name  # 日志文件名称
        self._log_level = log_level  # 日志等级
        self.__istrained = False

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
        self._optimizer_name = optimizer  # 优化器名称
        # 学习率调整策略
        if self._optimizer_name == "SGD":
            if optimizer_params is None:
                optimizer_params = {}
            print(optimizer_params)
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
            train_dataset.dataframe, train_dataset.x, train_dataset.y).params  # 最小二乘法获得线性回归系数
        print(self._weight)
        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # 用于将权重、系数、参数相乘的线性层
        self._out.weight = nn.Parameter(torch.tensor([self._weight]).to(
            torch.float32), requires_grad=False)  # 定义out层权重
        self._criterion = nn.MSELoss()  # 损失函数
        self._trainLossList = []  # 记录训练过程中的Loss
        self._validLossList = []  # 记录验证过程中的Loss
        self._epoch = 0  # 当前的epoch数
        self._bestr2 = float('-inf')  # 目前最好情况的r2
        self._noUpdateEpoch = 0  # r2没有提高的连续epoch数
        self._modelName = "GNNWR_" + model_name  # 模型名称，用于保存模型
        self._modelSavePath = model_save_path  # 模型保存路径
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
        self._model.train()  # 设置为train模式
        train_loss = 0  # 记录当前epoch的Loss
        data_loader = self._train_dataset.dataloader  # 数据加载器
        maxindex = len(data_loader)  # 数据加载器的长度
        for index, (data, coef, label) in enumerate(data_loader):
            if self._use_gpu:
                data, coef, label = data.cuda(), coef.cuda(), label.cuda()
            # data, label = data.view(
            #     data.shape[0], -1), label.view(data.shape[0], -1)  # 设置数据形状
            self._optimizer.zero_grad()  # 梯度清零

            # 先通过网络输出预测权重，再将权重、参数、系数相乘
            output = self._out(self._model(data).mul(coef.to(torch.float32)))
            loss = self._criterion(output, label)  # 计算Loss
            loss.backward()  # 反向传播
            self._optimizer.step()  # 模型参数优化
            if isinstance(data, list):
                train_loss += loss.item() * data[0].size(0)
            else:
                train_loss += loss.item() * data.size(0)  # Loss求和

            # 打印进度条
            i = index + 1
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-50s] %d%%" % ('#' * int(i * 50.0 / maxindex), int(100.0 * i / maxindex)))
            sys.stdout.flush()

        train_loss /= self._train_dataset.datasize  # Loss取平均
        self._trainLossList.append(train_loss)  # 记录在List中

    def __valid(self):
        """
        validate the network
        """
        self._model.eval()  # 设置为验证模式
        val_loss = 0  # 记录当前epoch的Loss
        label_list = np.array([])  # 标准值
        out_list = np.array([])  # 预测值
        data_loader = self._valid_dataset.dataloader  # 数据加载器

        with torch.no_grad():  # 验证模式中不需要进行梯度更新
            for data, coef, label in data_loader:
                if self._use_gpu:
                    data, coef, label = data.cuda(), coef.cuda(), label.cuda()

                # 先通过网络输出预测权重，再将权重、参数、系数相乘
                weight = self._model(data)
                output = self._out(self._model(
                    data).mul(coef.to(torch.float32)))
                loss = self._criterion(output, label)  # 计算Loss
                out_list = np.append(
                    out_list, output.view(-1).cpu().detach().numpy())  # 将预测值加入List中
                label_list = np.append(
                    label_list, label.view(-1).cpu().numpy())  # 将标准值加入List中
                if isinstance(data, list):
                    val_loss += loss.item() * data[0].size(0)
                else:
                    val_loss += loss.item() * data.size(0)  # Loss求和
            val_loss /= len(self._valid_dataset)  # Loss取平均
            self._validLossList.append(val_loss)  # 记录在List中
            r2 = r2_score(label_list, out_list)  # 根据预测值和标准值计算R方
            if r2 > self._bestr2:  # 如果R方比目前最高R方要高，则保存当前模型
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
                    out_list, output.view(-1).cpu().detach().numpy())  # 将预测值加入List中
                label_list = np.append(
                    label_list, label.view(-1).cpu().numpy())  # 将标准值加入List中
                if isinstance(data, list):
                    test_loss += loss.item() * data[0].size(0)
                else:
                    test_loss += loss.item() * data.size(0)  # Loss求和
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
            self._model = nn.DataParallel(module=self._model)  # 并行运算
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
            if 0 < early_stop < self._noUpdateEpoch:  # 如果达到早停标准则停止
                break
        self.__test()
        print("Test Loss: ", self.__testLoss, " Test R2: ", self.__testr2)
        logging.info("Test Loss: " + str(self.__testLoss) + " Test R2: " + str(self.__testr2))

    def predict(self, data_loader):
        # 23.6.8_TODO:load_model
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
                 log_file_name: str = "gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
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

        self._model = nn.Sequential(STPNN(dense_layers[0], 2, self._STPNN_out, drop_out, activate_func, batch_norm),
                                    SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                         activate_func, batch_norm))
        print(self._model)
