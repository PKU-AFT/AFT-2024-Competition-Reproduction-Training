import yaml
import pickle
import numpy as np
import pandas as pd
import torch

from dataloader import TSDataSampler, DailyBatchSamplerRandom
from rnnmodel import RNNModel

def main():
    # 1) 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["data_path"]
    drop_cols = config["data"]["drop_columns"]
    target_col = config["data"]["target_col"]

    # RNN/LSTM 模型参数
    model_type = config["model"]["model_type"]  # "RNN" 或 "LSTM"
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    n_epoch = config["model"]["n_epoch"]
    lr = config["model"]["lr"]
    GPU = config["model"]["GPU"]
    seed = config["model"]["seed"]
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
    print("Data columns:", data.columns.values)

    # 以 time_id 为例
    max_time_id = data.index.get_level_values('time_id').max()
    print("max_time_id", max_time_id)

    # 3) 数据集切分
    split_time_id = int(max_time_id * 0.8)
    train_data = data.query('time_id <= @split_time_id').copy()
    valid_data = data.query('time_id > @split_time_id').copy()
    del data

    # 4) 构建 TSDataSampler
    train_dataset = TSDataSampler(
        data=train_data,
        start=0,
        end=split_time_id,
        step_len=10,
        fillna_type='ffill+bfill',
    )
    valid_dataset = TSDataSampler(
        data=valid_data,
        start=split_time_id+1,
        end=max_time_id,
        step_len=10,
        fillna_type='ffill+bfill',
    )
    if train_dataset.empty or valid_dataset.empty:
        print("Dataset is empty, please check your data or config.")
        return

    # 5) 实例化 RNNModel (或者 LSTMModel)
    #    注意：这里 input_size=124 是因为 F=124-1 里66% etc...
    rnn_model = RNNModel(
        model_type=model_type,    # "RNN" or "LSTM"
        input_size=124,           # 你的特征维度
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_epochs=n_epoch,
        lr=lr,
        GPU=GPU,
        seed=seed,
        save_path=save_path,
        save_prefix=save_prefix
    )

    # 6) 训练
    rnn_model.fit(train_dataset, valid_dataset)
    print("RNN/LSTM Model Trained.")

    # 7) 测试 predict (和 MASTER示例一样，示例 dl_test 未定义)
    dl_test = None  # 需要你自行构造 TSDataSampler
    if dl_test is not None:
        predictions, metrics = rnn_model.predict(dl_test)
        print("Test predictions: ", predictions)
        print("Test metrics: ", metrics)
    else:
        print("dl_test is not defined. Provide a TSDataSampler for testing if needed.")

if __name__ == "__main__":
    main()
