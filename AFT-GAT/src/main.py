import yaml
import torch
import numpy as np

from dataloader import TSDataSampler, DailyBatchSamplerRandom
from train import train_gat_model
from tcnmodel import test_run  # 演示调用GAT内置测试
from basemodel import SequenceModel2

def main():
    # 1) 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    d_feat = config["model"]["d_feat"]
    hidden_size = config["model"]["hidden_size"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]
    base_model = config["model"]["base_model"]

    n_epochs = config["train"]["n_epochs"]
    lr = config["train"]["lr"]
    save_path = config["train"]["save_path"]
    save_prefix = config["train"]["save_prefix"]
    seed = config["train"]["seed"]

    step_len = config["data"]["step_len"]
    start_time = config["data"]["start_time"]
    end_time = config["data"]["end_time"]

    # 2) 演示：我们没有真实的数据，这里随机生成一个 DataFrame 做示例
    idx = pd.MultiIndex.from_product(
        [range(start_time, end_time), ["stock1", "stock2"]], names=["time_id", "stock_id"]
    )
    # 每行 70 个特征 + 1 个目标列 => 71 列
    df = pd.DataFrame(np.random.randn(len(idx), d_feat+1), index=idx)

    # 构造 TSDataSampler
    train_data = TSDataSampler(
        data=df,
        start=start_time,
        end=int(end_time * 0.8),
        step_len=step_len,
        fillna_type="ffill+bfill"
    )
    valid_data = TSDataSampler(
        data=df,
        start=int(end_time * 0.8) + 1,
        end=end_time,
        step_len=step_len,
        fillna_type="ffill+bfill"
    )

    if train_data.empty or valid_data.empty:
        print("Dataset is empty, please check your data range or input.")
        return

    print(f"Train sampler size: {len(train_data)}")
    print(f"Valid sampler size: {len(valid_data)}")

    # 构造采样器(可选)
    train_sampler = DailyBatchSamplerRandom(train_data, shuffle=True)
    valid_sampler = DailyBatchSamplerRandom(valid_data, shuffle=False)

    # 3) 调用封装好的训练函数
    print("===== Testing built-in GAT run from tcnmodel.py =====")
    test_out = test_run()  # 这是你原来的 GAT_TS2 测试

    print("\n===== Now training GAT model in a SequenceModel2 manner =====")
    trainer, best_loss = train_gat_model(
        train_dataset=train_data,
        valid_dataset=valid_data,
        n_epochs=n_epochs,
        lr=lr,
        save_path=save_path,
        save_prefix=save_prefix,
        seed=seed
    )
    print("Training completed, best validation loss = ", best_loss)

    # 4) 如需加载模型参数
    # param_path = f"{save_path}{save_prefix}epoch_0.pkl"
    # trainer.load_param(param_path)

    print("All done.")

if __name__ == "__main__":
    main()
