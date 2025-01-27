import torch
import numpy as np

from basemodel import SequenceModel2
from GAT import GAT_TS1, GAT_TS2, GAT, GAT_CS, FFN

def train_gat_model(
    train_dataset,
    valid_dataset,
    model_class=GAT_TS1,
    n_epochs=20,
    lr=0.001,
    save_path="./",
    save_prefix="gat_master_",
    seed=42
):
    """
    演示：使用 SequenceModel2 进行包装训练的流程。
    你可以替换/改写 loss_fn、数据结构等细节。
    """
    # 定义一个 SequenceModel2 子类，或者直接实例化 SequenceModel2
    # 并指定 model_class 作为其中的 self.model
    class GATSequenceModel(SequenceModel2):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 根据 model_class 构造真正的模型
            # 这里示例写固定超参，若要灵活，可改为从kwargs获取
            self.model = model_class(d_feat=70, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU")
            self.init_model()

        # 如果需要自定义 loss_fn，可在此处 override
        # def loss_fn(self, pred, label):
        #     return torch.mean((pred - label) ** 2)

    # 实例化并训练
    gat_trainer = GATSequenceModel(n_epochs=n_epochs, lr=lr, save_path=save_path, save_prefix=save_prefix, seed=seed)
    best_loss = gat_trainer.fit(train_dataset, valid_dataset)
    return gat_trainer, best_loss


# 如果需要更多训练流程，也可在这里添加
