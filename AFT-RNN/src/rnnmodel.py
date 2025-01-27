import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from basemodel import SequenceModel

class SimpleRNNNetwork(nn.Module):
    """
    一个示例的 RNN/LSTM 网络:
    - 输入 x.shape: (N, T, F)
    - 经过 RNN 或 LSTM 后取最后一个时间步
    - 最后线性映射到 1维输出
    """
    def __init__(self, input_size=124, hidden_size=64, num_layers=2, dropout=0.5, model_type="RNN"):
        super(SimpleRNNNetwork, self).__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if model_type.upper() == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        elif model_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported model_type={model_type}, must be 'RNN' or 'LSTM'.")

        # 最后一层
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [N, T, F]
        """
        # RNN输出: out.shape = [N, T, hidden_size]
        # LSTM还会返回 (h, c)
        if self.model_type.upper() == "RNN":
            out, h = self.rnn(x)  # out: [N, T, hidden_size]
            # 取最后一个时间步
            final_out = out[:, -1, :]  # [N, hidden_size]
        else:  # LSTM
            out, (h, c) = self.rnn(x)
            final_out = out[:, -1, :]  # [N, hidden_size]

        out = self.fc_out(final_out).squeeze(-1)  # => [N]
        # 如果需要对输出做其他处理，可在此添加
        return out

class RNNModel(SequenceModel):
    """
    继承自 SequenceModel, 在 init_model 中初始化 SimpleRNNNetwork.
    """
    def __init__(self,
                 model_type="RNN",
                 input_size=124,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.5,
                 **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.init_model()

    def init_model(self):
        self.model = SimpleRNNNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type=self.model_type
        )
        super(RNNModel, self).init_model()
