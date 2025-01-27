import copy
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

# 示例: 提前停机制
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


# 示例: 通用序列模型基类
class SequenceModel2:
    def __init__(self, n_epochs=20, lr=0.001, save_path="./", save_prefix="gat_master_", seed=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.fitted = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.model = None
        self.train_optimizer = None
        self.scheduler = None

    def load_param(self, param_path):
        if self.model is None:
            raise RuntimeError("Model is not defined yet!")
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def _init_data_loader(self, dataset, batch_sampler=None, shuffle=True, drop_last=True):
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler)
        else:
            return DataLoader(dataset, shuffle=shuffle, drop_last=drop_last)

    def init_model(self):
        """
        在子类中定义 self.model 之后再调用此函数初始化优化器等
        """
        if self.model is None:
            raise ValueError("Please define self.model before init_model.")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.train_optimizer, mode='min', factor=0.1, patience=5)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        # 默认使用 MSE 作为示例
        return torch.mean((pred - label) ** 2)

    def train_epoch(self, dataloader):
        self.model.train()
        losses = []
        for data in dataloader:
            # data: [batch_size, ...], 具体视数据而定
            # 这里仅示例：假设最后一列是label
            data = data.to(self.device).float()
            features = data[..., :-1]
            label = data[..., -1]
            pred = self.model(features)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            self.train_optimizer.step()

            losses.append(loss.item())
        return float(np.mean(losses))

    def test_epoch(self, dataloader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device).float()
                features = data[..., :-1]
                label = data[..., -1]
                pred = self.model(features)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
        return float(np.mean(losses))

    def fit(self, train_dataset, valid_dataset, train_batch_sampler=None, valid_batch_sampler=None):
        train_loader = self._init_data_loader(train_dataset, batch_sampler=train_batch_sampler, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(valid_dataset, batch_sampler=valid_batch_sampler, shuffle=False, drop_last=False)

        self.fitted = True
        best_param = None
        best_val_loss = float("inf")

        early_stopper = EarlyStopping(patience=10)

        for epoch_i in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print(f"Epoch {epoch_i}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_param = copy.deepcopy(self.model.state_dict())

            self.scheduler.step(val_loss)
            if early_stopper(val_loss):
                print("Early stopping triggered.")
                break

            # 存储当前 epoch 的最好模型
            torch.save(best_param, f"{self.save_path}{self.save_prefix}epoch_{epoch_i}.pkl")

        return best_val_loss
