import yaml
import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data, TestDataLoader, predict
from basemodel import EarlyStopper
from tcnmodel import build_model
from train import train_model, History, show_demo_plots


def main():
    # 1) 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hyper = config["hyperparams"]
    model_cfg = config["model"]
    paths = config["paths"]

    # 2) 加载数据 (示例)
    (X_train, y_train_main, y_train_aux), (X_valid, y_valid_main, y_valid_aux) = load_data(
        paths["train_data_path"], paths["valid_data_path"]
    )

    print("Train shape:", X_train.shape, "Valid shape:", X_valid.shape)
    # 构造符合 train_model 需要的结构
    train_dataset = (X_train, [y_train_main, y_train_aux])
    valid_dataset = (X_valid, [y_valid_main, y_valid_aux])

    # 3) 调用 train_model
    predictions = train_model(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_models=hyper["NUM_MODELS"],
        build_model_func=build_model,             # 直接使用 tcnmodel.py 的 build_model
        batch_size=hyper["BATCH_SIZE"],
        epochs=hyper["EPOCHS"],
        loss_weight=model_cfg["loss_weight"]
    )

    print("Validation predictions shape:", predictions.shape)
    print("Done training.")

    # 演示：构造一个 History 对象，给出一些伪数据
    dummy_history = History(
        epoch=[0,1,2,3],
        train_loss=[0.5, 0.4, 0.35, 0.3],
        test_loss=[0.48, 0.45, 0.4, 0.38],
        lr=[0.002, 0.0012, 0.00072, 0.000432]
    )

    # 4) 显示演示图
    # 如果你有真正的 history 数据，可以传入
    show_demo_plots(dummy_history, None, None)

    # 5) 如需推理，可以构造 test_dataloader
    # X_test = np.random.randn(100, X_train.shape[1], X_train.shape[2])  # 示例
    # test_dl = TestDataLoader(X_test, batch_size=32)
    # 这里就可以调用 predict(model, test_dl) 得到推理结果
    # ...

    print("All done.")

if __name__ == "__main__":
    main()
