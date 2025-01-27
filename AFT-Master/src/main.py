import yaml
import pickle
import numpy as np
import pandas as pd
import torch

from dataloader import TSDataSampler, DailyBatchSamplerRandom
from mastermodel import MASTERModel
from torch.utils.data import DataLoader

def main():
    # 1) 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["data_path"]
    drop_cols = config["data"]["drop_columns"]
    target_col = config["data"]["target_col"]

    d_feat = config["model"]["d_feat"]
    d_model = config["model"]["d_model"]
    t_nhead = config["model"]["t_nhead"]
    s_nhead = config["model"]["s_nhead"]
    dropout = config["model"]["dropout"]
    n_epoch = config["model"]["n_epoch"]
    lr = config["model"]["lr"]
    GPU = config["model"]["GPU"]
    seed = config["model"]["seed"]
    universe = config["model"]["universe"]
    save_path = config["model"]["save_path"]
    save_prefix = config["model"]["save_prefix"]

    # 2) 加载 data_for_model.pkl
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    data = data.drop(drop_cols, axis=1)
    # 调整列顺序，将 target 移到最后
    cols = list(data.columns)
    cols.remove(target_col)
    cols.append(target_col)
    data = data[cols]

    data = data.astype(np.float32)
    print(data.columns.values)

    max_time_id = data.index.get_level_values('time_id').max()
    print("max_time_id", max_time_id)

    # 3) 数据集切分
    split_time_id = int(max_time_id * 0.8)
    train_data = data.query('time_id <= @split_time_id').copy()
    valid_data = data.query('time_id > @split_time_id').copy()
    del data

    # 构建 TSDataSampler
    train_dataset = TSDataSampler(
        data=train_data, 
        start=0, 
        end=split_time_id, 
        step_len=10, 
        fillna_type='ffill+bfill'
    )
    valid_dataset = TSDataSampler(
        data=valid_data, 
        start=split_time_id+1, 
        end=max_time_id, 
        step_len=10, 
        fillna_type='ffill+bfill'
    )

    # 构建 DataLoader
    train_sampler = DailyBatchSamplerRandom(train_dataset, shuffle=False)
    valid_sampler = DailyBatchSamplerRandom(valid_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler)

    # 4) 初始化 MASTERModel
    model = MASTERModel(
        d_feat=d_feat, 
        d_model=d_model, 
        t_nhead=t_nhead, 
        s_nhead=s_nhead, 
        T_dropout_rate=dropout, 
        S_dropout_rate=dropout,
        n_epochs=n_epoch, 
        lr=lr, 
        GPU=GPU, 
        seed=seed,
        save_path=save_path, 
        save_prefix=save_prefix
    )

    # 5) 训练
    model.fit(train_dataset, valid_dataset)
    print("Model Trained.")

    # 6) 测试阶段(原代码中出现 dl_test，但没有定义; 这里保持原样，不做省略)
    # 你需要自行定义 dl_test 或修改逻辑
    # 假设 dl_test 是一个 TSDataSampler
    dl_test = None  # 请根据需要替换成实际数据
    if dl_test is not None:
        predictions, metrics = model.predict(dl_test)
        print(metrics)
    else:
        print("dl_test is not defined. Please define your test dataset if needed.")

if __name__ == "__main__":
    main()
