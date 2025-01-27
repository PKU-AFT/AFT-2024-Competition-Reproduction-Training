import os
import gc
import pickle
import numpy as np
import pandas as pd

def load_data(train_data_path, valid_data_path):
    """
    示例：从 pickle 文件中读取训练集/验证集数据。
    这里只是演示，请根据你的实际需要实现。
    """
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    
    # 假设 train_data, valid_data 里已经包含 (X, y_main, y_aux)
    X_train, y_train_main, y_train_aux = train_data
    X_valid, y_valid_main, y_valid_aux = valid_data

    return (X_train, y_train_main, y_train_aux), (X_valid, y_valid_main, y_valid_aux)


class TestDataLoader:
    """
    一个示例的 test dataloader，假设用于推理。
    你可根据需要改写。
    """
    def __init__(self, X_test, batch_size=32):
        self.X_test = X_test
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.X_test):
            raise StopIteration
        batch_X = self.X_test[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch_X


def predict(model, data_loader):
    """
    演示：从自定义迭代器中 batch 读取数据进行推理。
    """
    preds = []
    for batch_X in data_loader:
        # 注意：这里是Keras模型，因此无需 .to(device)
        batch_pred = model.predict(batch_X)
        preds.extend(batch_pred.flatten())
    return np.array(preds)
