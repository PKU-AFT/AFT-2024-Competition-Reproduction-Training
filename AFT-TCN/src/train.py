# train.py

import numpy as np
import optuna
import torch
import matplotlib.pyplot as plt

from tcnmodel import TCNSequenceModel
from dataloader import TSDataSampler
# 如果要用 show 图，需要在某些环境下使用非 inline 模式
# %matplotlib inline  # <- Jupyter 魔法命令，已注释

def objective(trial, train_dataset, valid_dataset, feature_dim):
    """
    Optuna 目标函数，用于超参搜索
    """
    # 获取超参数
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2) 
    batch_size = trial.suggest_int('batch_size', 16, 128, step=32)
    n_channels_0 = trial.suggest_int('n_channels_0', 16, 64, step=16)
    n_channels_1 = trial.suggest_int('n_channels_1', 32, 128, step=32)
    n_channels_2 = trial.suggest_int('n_channels_2', 64, 256, step=64)
    num_channels = [n_channels_0, n_channels_1, n_channels_2]
    
    # 定义模型
    model = TCNSequenceModel(
        n_epochs=20,  # 也可放到 trial 超参中
        lr=lr,
        input_size=feature_dim,
        output_size=1,
        num_channels=num_channels,
        kernel_size=2,
        dropout=0.2,
    )
    
    # 训练模型
    train_losses, val_losses = model.fit(train_dataset, valid_dataset)
    # 返回最后一个 epoch 的验证集损失
    return val_losses[-1]

def run_optuna_study(train_dataset, valid_dataset, config):
    """
    运行 optuna 搜索
    """
    optuna_cfg = config["optuna"]
    n_trials = optuna_cfg["n_trials"]

    # 这里使用 YAML 中 data.feature_dim 作为 input_size
    feature_dim = config["data"]["feature_dim"]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataset, valid_dataset, feature_dim), n_trials=n_trials)
    
    return study

def train_final_model(train_dataset, valid_dataset, config):
    """
    使用 config 中的参数训练最终模型并可视化损失
    """
    train_cfg = config["train"]
    data_cfg = config["data"]

    # 实例化 TCNSequenceModel
    model = TCNSequenceModel(
        n_epochs=train_cfg["n_epochs"],
        lr=train_cfg["lr"],
        input_size=data_cfg["feature_dim"],
        output_size=train_cfg["output_size"],
        num_channels=train_cfg["num_channels"],
        kernel_size=train_cfg["kernel_size"],
        dropout=train_cfg["dropout"],
        seed=train_cfg.get("seed", None),
        save_path=train_cfg.get("save_path", "/kaggle/working/"),
        save_prefix=train_cfg.get("save_prefix", "master_")
    )

    # 训练模型
    train_losses, val_losses = model.fit(train_dataset, valid_dataset)

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


def load_and_finetune_model(model, param_path, train_dataset, valid_dataset):
    """
    加载已有模型参数并再训练
    """
    model.load_param(param_path)
    train_losses, val_losses = model.fit(train_dataset, valid_dataset)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (After Loading Param)')
    plt.legend()
    plt.tight_layout()
    plt.show()
