import math

import torch
import torch.nn as nn


def default_dense_layer(insize, outsize):
    dense_layer = []
    size = int(math.pow(2, int(math.log2(insize))))
    while size > outsize:
        dense_layer.append(size)
        size = int(math.pow(2, int(math.log2(size)) - 1))
    return dense_layer


class SWNN(nn.Module):
    # dense_layer：全连接层大小,insize：输入层大小,outsize：输出层大小，drop_out默认0.2,activate_func：激活函数，默认为PRelu(0.4),batch_norm：是否使用批归一化层
    def __init__(self, dense_layer=None, insize=-1, outsize=-1, drop_out=0.2, activate_func=nn.PReLU(init=0.4),
                 batch_norm=True):
        super(SWNN, self).__init__()
        if dense_layer is None or len(dense_layer) == 0:
            self.dense_layer = default_dense_layer(insize, outsize)
        else:
            self.dense_layer = dense_layer
        if insize < 0 or outsize < 0:
            raise ValueError("insize and outsize must be positive")
        self.drop_out = drop_out
        self.batch_norm = batch_norm
        self.activate_func = activate_func
        self.insize = insize
        self.outsize = outsize
        count = 0  # 用于层命名
        lastsize = self.insize  # 用于记录上一层的size
        self.fc = nn.Sequential()
        for size in self.dense_layer:
            self.fc.add_module("swnn_full" + str(count),
                               nn.Linear(lastsize, size))  # 添加全连接层
            if batch_norm:
                # 如果需要批归一化则添加批归一化层
                self.fc.add_module("swnn_batc" + str(count), nn.BatchNorm1d(size))
            self.fc.add_module("swnn_acti" + str(count), self.activate_func)  # 添加激活函数
            self.fc.add_module("swnn_drop" + str(count),
                               nn.Dropout(self.drop_out))  # 添加drop_out层
            lastsize = size  # 更新上一层size
            count += 1
        self.fc.add_module("full" + str(count),
                           nn.Linear(lastsize, self.outsize))  # 连接最后一个隐藏层与输出层

    def forward(self, x):
        x.to(torch.float32)
        x = self.fc(x)
        return x


class STPNN(nn.Module):
    def __init__(self, dense_layer, insize, outsize, drop_out=0.2, activate_func=nn.PReLU(init=0.4), batch_norm=False):
        super(STPNN, self).__init__()
        # dense_layer的默认值
        self.dense_layer = dense_layer
        self.drop_out = drop_out
        self.batch_norm = batch_norm
        self.activate_func = activate_func
        self.insize = insize
        self.outsize = outsize
        count = 0  # 用于层命名
        lastsize = self.insize  # 用于记录上一层的size
        self.fc = nn.Sequential()
        for size in self.dense_layer:
            self.fc.add_module("stpnn_full" + str(count),
                               nn.Linear(lastsize, size))  # 添加全连接层
            if batch_norm:
                # 如果需要批归一化则添加批归一化层
                self.fc.add_module("stpnn_batc" + str(count), nn.BatchNorm1d(size))
            self.fc.add_module("stpnn_acti" + str(count), self.activate_func)  # 添加激活函数
            self.fc.add_module("stpnn_drop" + str(count),
                               nn.Dropout(self.drop_out))  # 添加drop_out层
            lastsize = size  # 更新上一层size
            count += 1
        self.fc.add_module("full" + str(count), nn.Linear(lastsize, self.outsize))  # 连接最后一个隐藏层与输出层

    def forward(self, x):
        if isinstance(x, list):
            spatial_x, temporal_x = x
            spatial_x = spatial_x.squeeze()
            temporal_x = temporal_x.squeeze()
            # TPNN
            temporal_model = nn.Sequential(torch.nn.Linear(temporal_x.shape[-1], self.outsize),nn.PReLU(init=0.4))
            temporal_output = weight_share(temporal_model, temporal_x,self.outsize)

            # SPNN
            spatial_model = nn.Sequential(torch.nn.Linear(spatial_x.shape[-1], self.outsize),nn.PReLU(init=0.4))
            spatial_output = weight_share(spatial_model, spatial_x, self.outsize)
            x = torch.cat((spatial_output, temporal_output), dim=2)
        # STPNN
        output = weight_share(self.fc, x, self.outsize)
        output = torch.reshape(output, shape=(output.shape[0], output.shape[1] * output.shape[2]))
        return output


# 权共享计算
def weight_share(model, x, output_size=1):
    x.to(torch.float32)
    batch = x.shape[0]
    height = x.shape[1]
    x = torch.reshape(x, shape=(batch * height, x.shape[2]))
    output = model(x)
    output = torch.reshape(output, shape=(batch, height, output_size))
    return output
