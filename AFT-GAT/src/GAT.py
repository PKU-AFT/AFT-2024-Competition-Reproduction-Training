import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.name = "FFN"

    def forward(self, x):
        hidden = self.fc(x)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()

class GAT_CS(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.num_layers = num_layers
        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)
        self.name = "GAT_CS"

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)
        batch_size = x.shape[0]
        sample_num = x.shape[1]
        dim = x.shape[2]
        x = x.unsqueeze(1)
        e_x = x.expand(batch_size, sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 1, 2)
        attention_in = torch.cat((e_x, e_y), 3).view(batch_size, -1, dim * 2)
        attention_out = attention_in.matmul(self.a).view(batch_size, sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x, mask=None):
        hidden = self.fc_in(x)
        for _ in range(self.num_layers):
            att_weight = self.cal_attention(hidden, hidden)
            hidden = torch.einsum("bas,bsf->baf", att_weight, hidden) + hidden
        return hidden # [batch_size, stocks, hidden_size]

class GAT(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.gat_cs = GAT_CS(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers)
        self.fc_out = FFN(hidden_size=hidden_size)
        self.name = "GAT"
    
    def forward(self, x, mask=None):
        hidden = self.gat_cs(x)
        return self.fc_out(hidden)

class GAT_TS1(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        
        self.gat = GAT_CS(d_feat=hidden_size, hidden_size=hidden_size)
        self.fc_out = FFN(hidden_size=hidden_size)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.name = "GAT_TS1"

    def forward(self, x, mask=None):
        # x: [batch, stocks, lookback, features]
        x = x.permute(0, 2, 1, 3)  # => [batch, stocks, lookback, features] => [batch, stocks, lookback, features] 
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3])  # => [batch * stocks, lookback, features]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # => [batch * stocks, hidden_size]
        hidden = hidden.reshape(batch_size, -1, self.hidden_size)  # => [batch, stocks, hidden_size]
        hidden = self.gat(hidden)  # => [batch, stocks, hidden_size]
        out = self.fc_out(hidden)  # => [batch, stocks]
        return out

class GAT_TS2(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, base_model="GRU"):
        super().__init__()
        if base_model == "GRU":
            self.rnn_cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        elif base_model == "LSTM":
            # 原代码里写的是 self.rnn_cell = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
            self.rnn_cell = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.hidden_size = hidden_size
        self.fc_out = FFN(hidden_size=hidden_size)
        self.gat = GAT_CS(d_feat=hidden_size, hidden_size=hidden_size)
        self.name = "GAT_TS2"

    def forward(self, x, mask=None):
        # x: [batch, stocks, lookback, features]
        x = x.permute(0, 2, 1, 3)
        batch_size = x.shape[0]
        x = self.fc_in(x)  # => [batch, stocks, lookback, hidden_size]
        hidden = None
        for i in range(x.shape[2] - 1):
            if hidden is not None:
                out = hidden.view(batch_size, -1, self.hidden_size)
                out = self.gat(out) 
                out = out.view(-1, self.hidden_size)
                input = x[:, :, i, :].view(-1, self.hidden_size)
                # rnn_cell
                if isinstance(self.rnn_cell, nn.GRUCell):
                    out = self.rnn_cell(input, out)
                    hidden = out
                else:
                    h, c = hidden
                    h = self.rnn_cell(input, (out, c))[0]
                    hidden = (h, c)
            else:
                # 第一步
                input = x[:, :, 0, :]
                out = self.gat(input)  # [batch_size, stocks, hidden_size]
                out = out.view(-1, self.hidden_size)
                if isinstance(self.rnn_cell, nn.GRUCell):
                    out = self.rnn_cell(out)
                    hidden = out
                else:
                    # LSTMCell 返回 (h, c)
                    hidden = self.rnn_cell(out)
        # 最终 hidden
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        res = hidden.view(batch_size, -1, self.hidden_size)
        return self.fc_out(res) # [batch_size, stocks]


def test_run():
    print("Running test with random input...")
    input_data = torch.randn(100, 30, 16, 70)
    # model = GAT_CS(d_feat=70, hidden_size=64, num_layers=2)
    model = GAT_TS2(d_feat=70, hidden_size=64)
    output = model(input_data)
    print("Output shape:", output.shape)
    return output

if __name__ == "__main__":
    test_out = test_run()
    # Output shape => [100, 30]
    print("Test run completed, output shape is", test_out.shape)
