# main.py

import os
import yaml
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 如果你不想在终端输出不必要的警告，可取消注释
# warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)

# %matplotlib inline  # <- Jupyter 魔法命令，已注释，但可保留以示原本功能

from dataloader import TSDataSampler
from train import run_optuna_study, train_final_model, load_and_finetune_model

def main():
    # 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 从 config 中读取关键信息
    parquet_path = config["data"]["parquet_path"]
    drop_cols = config["data"]["drop_columns"]
    feature_dim = config["data"]["feature_dim"]
    step_len = config["data"]["step_len"]
    split_ratio = config["data"]["split_ratio"]

    # 加载数据
    print(f"Reading parquet file from: {parquet_path}")
    with open(parquet_path, 'rb') as f:
        data = pd.read_parquet(f)

    # 丢弃指定列
    data = data.drop(drop_cols, axis=1)

    # 调整列顺序，将 target 移动到最后
    cols = list(data.columns)
    if 'target' in cols:
        cols.remove('target')
        cols.append('target')
    data = data[cols]

    # 转 float32
    data = data.astype(np.float32)

    print("Data columns:", data.columns.values)

    # 以 (time_id, stock_id) 作为索引
    data = data.set_index(['time_id', 'stock_id'])
    max_time_id = data.index.get_level_values('time_id').max()
    split_time_id = int(max_time_id * split_ratio)

    print("max_time_id:", max_time_id)
    print("split_time_id:", split_time_id)

    # 拆分数据集
    train_data = data.query('time_id <= @split_time_id').copy()
    valid_data = data.query('time_id > @split_time_id').copy()

    del data

    # 构建 TSDataSampler
    train_dataset = TSDataSampler(
        data=train_data,
        start=0,
        end=split_time_id,
        step_len=step_len,
        fillna_type='ffill+bfill'
    )
    valid_dataset = TSDataSampler(
        data=valid_data,
        start=split_time_id + 1,
        end=max_time_id,
        step_len=step_len,
        fillna_type='ffill+bfill'
    )

    if train_dataset.empty or valid_dataset.empty:
        print("Dataset is empty, please check your data or config.")
        return

    # -------------------
    # 1) 简单训练并可视化
    # -------------------
    print("Start training final model...")
    model = train_final_model(train_dataset, valid_dataset, config)

    # -------------------
    # 2) 进行 Optuna 超参搜索
    # -------------------
    print("Start Optuna study...")
    study = run_optuna_study(train_dataset, valid_dataset, config)
    print(f"Best trial: {study.best_trial.params}")

    # -------------------
    # 3) 演示加载已有参数并继续训练
    # -------------------
    # 假设你之前保存的某个权重:
    param_path = os.path.join(config["train"]["save_path"], "master_15.pkl")
    if os.path.exists(param_path):
        print(f"Loading param from {param_path} and fine-tuning...")
        load_and_finetune_model(model, param_path, train_dataset, valid_dataset)
    else:
        print(f"Param file {param_path} not found, skip loading.")

    # 如果需要，你也可以在此处做更多事情，比如测试集评估、可视化等
    print("All done.")


if __name__ == "__main__":
    main()
