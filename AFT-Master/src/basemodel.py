import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import DailyBatchSamplerRandom


class SequenceModel:
    """
    原代码中的 SequenceModel, 保留所有方法和属性。
    """
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path='model/', save_prefix=''):
        self.n_epochs = n_epochs
        self.lr = lr
        # 在原代码中 device 写死成 "mps:0"；这里保留原逻辑，但可扩展一下
        if GPU is not None and isinstance(GPU, int):
            device_str = f"cuda:{GPU}"
        else:
            # 如果需要使用mps请改为 "mps:0"
            device_str = "mps:0"
        self.device = torch.device(device_str)

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

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = torch.abs(pred[mask] - label[mask])
        return torch.mean(loss)

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
        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
        return float(np.mean(losses))

    def _init_data_loader(self, dataset, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(dataset, shuffle)
        data_loader = DataLoader(dataset, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dl_train, dl_valid):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            best_param = copy.deepcopy(self.model.state_dict())
            torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{step}.pkl')

    def predict(self, dl_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)
        preds = []
        labels = []
        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())
            labels.append(label.numpy().ravel())
        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        return predictions, labels
