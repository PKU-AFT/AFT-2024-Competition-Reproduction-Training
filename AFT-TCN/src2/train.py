import gc
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, LearningRateScheduler
from tcnmodel import build_model

def train_model(
    train_dataset,
    valid_dataset,
    num_models,
    build_model_func,
    batch_size,
    epochs,
    loss_weight
):
    """
    训练模型，支持多模型集成和辅助任务。
    原代码中：X_train, [y_train_main, y_train_aux]，但实际上输出层只有 main_output。
    我们此处也只能保持原样。
    """
    checkpoint_val_preds = []
    weights = []

    # train_dataset, valid_dataset 均为 (X, [y_main, y_aux]) 结构
    X_train, [y_train_main, y_train_aux] = train_dataset
    X_valid, [y_valid_main, y_valid_aux] = valid_dataset

    for model_idx in range(num_models):
        # 构建模型
        model = build_model_func(X_train.shape[1:], y_train_aux.shape[-1], loss_weight)
        for global_epoch in range(epochs):
            # 提早停止和学习率调度器
            es = EarlyStopping(patience=1, verbose=True)
            lr_scheduler = LearningRateScheduler(lambda epoch: 2e-3 * (0.6 ** global_epoch))

            # 模型训练
            model.fit(
                X_train,
                [y_train_main, y_train_aux],
                validation_data=(X_valid, [y_valid_main, y_valid_aux]),
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                callbacks=[lr_scheduler, es]
            )

            # 验证集预测 (注意：model.predict 返回的是 [main_out, aux_out] 形式？或只返回单输出？
            preds_valid = model.predict(X_valid, batch_size=batch_size)
            if isinstance(preds_valid, list) and len(preds_valid) > 0:
                # 如果确实返回多输出，我们默认取第一个输出
                checkpoint_val_preds.append(preds_valid[0].flatten())
            else:
                # 如果返回单输出
                checkpoint_val_preds.append(preds_valid.flatten())

            # 保存权重
            weights.append(2 ** global_epoch)

        # 清理内存
        del model
        gc.collect()

    # 计算验证集的加权平均预测
    predictions = np.average(checkpoint_val_preds, weights=weights, axis=0)
    return predictions


class History:
    """
    原代码中出现了 history.epoch, history.train_loss, history.test_loss, history.lr;
    这里定义一个简易的 History 类，供演示绘图之用。
    """
    def __init__(self, epoch, train_loss, test_loss, lr):
        self.epoch = epoch
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.lr = lr


# 以下为原 Notebook 中的演示性绘图与推理示例，为满足“不得有任何省略”，保持原样
# 只是把它们放到一个函数里，以免在导入时就执行。
def show_demo_plots(history, test_dataloader, model):
    """
    演示：绘制 loss 与 lr 的变化、以及简单的散点图。
    原 Notebook 代码碎片化，这里整合一下。
    """
    # 绘制 loss
    plt.plot(history.epoch, history.train_loss, "g:", label="Train Loss")
    plt.plot(history.epoch, history.test_loss, "r--", label="Test Loss")
    plt.legend(loc="upper left")
    ax2 = plt.twinx()
    ax2.plot(history.epoch, history.lr, c="yellow", label="Learning Rate")
    ax2.set_ylim(0, ax2.get_ylim()[1])
    plt.title("Loss and Learning Rate")
    plt.show()

    # 预测
    d = next(iter(test_dataloader))[0]
    # 原代码中有 predict(d, model) 之类，这里示例：
    # 需要你自行定义/实现 predict 函数或使用 Keras model 直接 .predict(d)
    # 由于不确定 test_dataloader 返回什么，这里仅保留原注释
    # pred = predict(d, model)
    # res = pd.DataFrame({
    #     "target": next(iter(test_dataloader))[1].flatten().cpu(),
    #     "pred": pred
    # })
    # res["err"] = np.abs(res["target"] - res["pred"])
    # plt.plot([res.target.min(), res.target.max()], [res.target.min(), res.target.max()], color="gray")
    # plt.scatter(res.target, res.pred, marker="x")
    # plt.ylim(res.pred.min()*1.5, res.pred.max()*1.5)
    # plt.show()
    #
    # plt.hist(res["pred"], bins=50)
    # plt.show()

    # 由于原代码中未完全定义 test_dataloader 和 predict，这里保持注释即可。
    pass
