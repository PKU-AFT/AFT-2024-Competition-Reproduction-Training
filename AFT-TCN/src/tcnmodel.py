import torch
import torch.nn as nn
import numpy as np
import copy
from src.basemodel import SequenceModel2


class Chomp1d(nn.Module):
    """
    用于移除卷积后序列末尾的padding
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    TCN的基本构建块，包含两层因果卷积
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, 
            self.chomp1, 
            self.relu1, 
            self.dropout1,
            self.conv2, 
            self.chomp2, 
            self.relu2, 
            self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    完整的TCN模型实现
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        # 转换为 TCN 期望的: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        
        # 通过 TCN 网络
        x = self.network(x)
        
        # 只取最后一个时间步的输出
        x = x[:, :, -1]
        
        # 最后一层线性层
        x = self.linear(x)
        return x


class TCNSequenceModel(SequenceModel2):
    """
    基于TCN模型的序列预测类
    """
    def __init__(self, n_epochs, lr, input_size, output_size=1, num_channels=[32, 64, 128], 
                 kernel_size=2, dropout=0.2, **kwargs):
        super(TCNSequenceModel, self).__init__(n_epochs, lr, **kwargs)
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # 定义具体的 TCNModel
        self.model = TCNModel(
            input_size=input_size,
            output_size=output_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 初始化优化器等
        self.init_model()

    def fit(self, dl_train, dl_valid):
        # 在训练前可检查输入特征维度是否匹配
        # (已在 SequenceModel2 中实现了 fit 方法，直接调用父类)
        train_losses, val_losses = super().fit(dl_train, dl_valid)
        return train_losses, val_losses
