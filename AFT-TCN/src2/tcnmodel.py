import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import (
    Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate,
    CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Lambda
)
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras import backend as K
from tqdm import tqdm_notebook as tqdm
import pickle
import gc

# 这里是你在 Notebook 中定义的常量，原样保留
BATCH_SIZE = 512
TCN_UNITS = 128
DENSE_HIDDEN_UNITS = 2 * TCN_UNITS
EPOCHS = 20
MAX_LEN = 220
NUM_MODELS = 1

# from https://github.com/philipperemy/keras-tcn
import keras.backend as K
import keras.layers
from keras import optimizers
# from keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer
from keras.utils.layer_utils import get_source_inputs
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import Convolution1D
from typing import List, Tuple

def channel_normalization(x):
    """ Normalize a layer to the maximum activation
    This keeps a layer's values between zero and one.
    """
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out

def wave_net_activation(x):
    """The activation used in WaveNet."""
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return keras.layers.multiply([tanh_out, sigm_out])

def residual_block(x, s, i, activation, nb_filters, kernel_size, padding, dropout_rate=0, name=''):
    """
    Defines the residual block for the WaveNet TCN.
    """
    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding=padding,
                  name=name + '_dilated_conv_%d_tanh_s%d' % (i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)

    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x

def process_dilations(dilations):
    """ 确保 dilations 为 2 的幂，否则自动转换。"""
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

class TCN(Layer):
    """
    Keras TCN layer (from https://github.com/philipperemy/keras-tcn).
    """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='norm_relu',
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=True,
                 name='tcn'):
        super().__init__()
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        if padding not in ['causal', 'same']:
            raise ValueError("Only 'causal' or 'same' paddings are compatible for this layer.")

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = Convolution1D(self.nb_filters, 1, padding=self.padding, name=self.name + '_initial_conv')(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x, s, i, self.activation, self.nb_filters,
                                             self.kernel_size, self.padding, self.dropout_rate, name=self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        x = Activation('relu')(x)

        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x


# 自定义损失函数
def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:, 0], (-1, 1)), y_pred) * y_true[:, 1]


# 构建模型函数
def build_model(input_shape, num_aux_targets, loss_weight, tcn_units=64, dense_hidden_units=128):
    """
    构建用于时序多特征数据的 Keras TCN 模型（双向思路：通过翻转输入序列来模拟反向）。
    原始代码中仅返回 main_output，但编译时却给了 2 个loss，对应 main_output + (aux_output? 已注释)。
    """
    inputs = Input(shape=input_shape)

    # 前向 TCN 层
    x1 = TCN(tcn_units, return_sequences=True, dilations=[1, 2, 4, 8, 16], name='tcn1_forward')(inputs)

    # 反向 TCN 层（通过翻转序列实现）
    x2 = Lambda(lambda z: K.reverse(z, axes=-2))(inputs)
    x2 = TCN(tcn_units, return_sequences=True, dilations=[1, 2, 4, 8, 16], name='tcn1_backward')(x2)

    # 合并前向和反向层
    x = add([x1, x2])

    # 第二层前向 TCN
    x1 = TCN(tcn_units, return_sequences=True, dilations=[1, 2, 4, 8, 16], name='tcn2_forward')(x)

    # 第二层反向 TCN
    x2 = Lambda(lambda z: K.reverse(z, axes=-2))(x)
    x2 = TCN(tcn_units, return_sequences=True, dilations=[1, 2, 4, 8, 16], name='tcn2_backward')(x2)

    # 再次合并
    x = add([x1, x2])

    # 全局池化
    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])

    # 全连接层 + 残差连接
    hidden = add([hidden, Dense(dense_hidden_units, activation='relu')(hidden)])
    hidden = add([hidden, Dense(dense_hidden_units, activation='relu')(hidden)])

    # 主任务输出
    main_output = Dense(1, activation='sigmoid', name='main_output')(hidden)
    
    # 辅助任务输出（原代码被注释掉）
    # aux_output = Dense(num_aux_targets, activation='sigmoid', name='aux_output')(hidden)

    # 只定义一个输出，却在compile中给了2个loss，原代码如此，原样保留
    model = Model(inputs=inputs, outputs=main_output)
    model.compile(
        loss=[custom_loss, 'binary_crossentropy'],
        loss_weights=[loss_weight, 1.0],
        optimizer=optimizers.Adam()
    )

    return model
