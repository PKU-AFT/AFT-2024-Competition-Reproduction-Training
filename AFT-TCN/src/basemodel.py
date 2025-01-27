import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 如果需要 fastdtw, spearmanr 等，可自行导入
# from fastdtw import fastdtw
# from scipy.stats import spearmanr

# 从 dataloader 中导入 DailyBatchSamplerRandom
from dataloader import DailyBatchSamplerRandom


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class SequenceModel2():
    """
    基础序列模型类，包含通用的训练流程、损失函数、early stopping 等。
    具体模型（如 TCN）可继承该类并在此基础上扩展。
    """
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, 
                 save_path='/kaggle/working/', save_prefix=''):
        self.n_epochs = n_epochs
        self.lr = lr
        # 这里根据你的需要自定义 device
        self.device = torch.device("cuda")  # 或者: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False
        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def init_model(self):
        """
        初始化模型及优化器
        """
        if self.model is None:
            raise ValueError("model 未定义，请在子类中设置 self.model！")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.train_optimizer, mode='min', factor=0.1, patience=10)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        """
        默认使用 L1 loss，可自行替换。
        """
        mask = ~torch.isnan(label)
        loss = torch.abs(pred[mask] - label[mask])
        return torch.mean(loss)

        # 示例：使用IC-based loss
        # mask = ~torch.isnan(label)
        # pred_np = pred[mask].detach().cpu().numpy()
        # label_np = label[mask].detach().cpu().numpy()
        # ic = spearmanr(pred_np, label_np).correlation
        # loss = 1 - ic
        # return torch.tensor(loss, dtype=torch.float32, requires_grad=True)

        # 示例：使用MSE
        # return torch.mean((pred - label) ** 2)

        # 示例：使用DTW
        # pred_np = pred.cpu().detach().numpy()
        # label_np = label.cpu().detach().numpy()
        # distance, _ = fastdtw(pred_np, label_np, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        # return torch.tensor(distance, requires_grad=True)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in data_loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].to(self.device)

                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

        return float(np.mean(losses))

    def fit(self, dl_train, dl_valid):
        """
        训练主流程
        """
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        train_losses = []
        val_losses = []

        early_stopping = EarlyStopping(patience=10)

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {step}, train_loss {train_loss:.6f}, valid_loss {val_loss:.6f}")
            best_param = copy.deepcopy(self.model.state_dict())

            self.scheduler.step(val_loss)

            if early_stopping(val_loss):
                print("Early stopping triggered.")
                break

            torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{step}.pkl')

        return train_losses, val_losses
