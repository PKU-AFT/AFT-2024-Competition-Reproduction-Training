# dataloader.py

import bisect
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, Union, List

import torch
from torch.utils.data import Sampler

# 这里原先有一条笔误的“devicefrom torch.utils.data import ...”命令，已修正为正常import
from torch.utils.data import DataLoader, TensorDataset, random_split


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    """
    如果索引不是排序好的，则对其进行排序。
    """
    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        and isinstance(idx, pd.MultiIndex)
        and not idx.is_lexsorted()
    ):
        return df.sort_index(axis=axis)
    else:
        return df


def np_ffill(arr: np.array):
    """
    用前值填充 NaN
    """
    mask = np.isnan(arr.astype(float))  # np.isnan只对float型有效
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


class TSDataSampler:
    """
    (T)ime-(S)eries DataSampler
    用于构建时间序列数据集（回看窗口模式），类似于 pytorch Dataset。
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        start, 
        end, 
        step_len: int, 
        fillna_type: str = "none", 
        dtype=None, 
        flt_data=None
    ):
        """
        data: 原始DataFrame数据，index=[(time_id, stock_id)], columns=[features..., target]
        start, end: 时间索引的起止，用于slice
        step_len: 时间窗口长度
        fillna_type: 缺失值填充方式: 'none', 'ffill', 'ffill+bfill'
        flt_data: 可选的过滤布尔序列
        """
        self.start = start
        self.end = end
        self.step_len = step_len
        self.fillna_type = fillna_type
        self.data = lazy_sort_index(data)

        kwargs = {"object": self.data}
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.data_arr = np.array(**kwargs)
        # 在末尾追加一行全NaN，用于padding
        self.data_arr = np.append(
            self.data_arr, 
            np.full((1, self.data_arr.shape[1]), np.nan, dtype=self.data_arr.dtype), 
            axis=0
        )
        self.nan_idx = -1  # 该行全NaN

        # 构建索引表
        self.idx_df, self.idx_map = self.build_index(self.data)
        self.data_index = deepcopy(self.data.index)

        if flt_data is not None:
            if isinstance(flt_data, pd.DataFrame):
                assert len(flt_data.columns) == 1
                flt_data = flt_data.iloc[:, 0]
            # bool(np.nan) 是 True，因此需要用 fillna(False) 来替换 NaN
            flt_data = flt_data.reindex(self.data_index).fillna(False).astype(np.bool_)
            self.flt_data = flt_data.values
            self.idx_map = self.flt_idx_map(self.flt_data, self.idx_map)
            self.data_index = self.data_index[np.where(self.flt_data)[0]]
        self.idx_map = self.idx_map2arr(self.idx_map)

        self.start_idx, self.end_idx = self.data_index.slice_locs(start=start, end=end)
        self.idx_arr = np.array(self.idx_df.values, dtype=np.float64)

        del self.data

    @staticmethod
    def idx_map2arr(idx_map):
        """
        将 idx_map 字典转换成 numpy 数组，避免在pytorch中使用过大的字典导致内存占用。
        """
        dtype = np.int64
        # 用最大值作为一个不存在的标记
        no_existing_idx = (np.iinfo(dtype).max, np.iinfo(dtype).max)

        max_idx = max(idx_map.keys())
        arr_map = []
        for i in range(max_idx + 1):
            arr_map.append(idx_map.get(i, no_existing_idx))
        arr_map = np.array(arr_map, dtype=dtype)
        return arr_map

    @staticmethod
    def flt_idx_map(flt_data, idx_map):
        idx = 0
        new_idx_map = {}
        for i, exist in enumerate(flt_data):
            if exist:
                new_idx_map[idx] = idx_map[i]
                idx += 1
        return new_idx_map

    def get_index(self):
        return self.data_index[self.start_idx: self.end_idx]

    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def build_index(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        将原始 data 的行索引拆分成 (i, j)，
        其中 i 表示日期层面行号， j 表示股票层面列号。
        """
        idx_df = pd.Series(range(data.shape[0]), index=data.index, dtype=object)
        idx_df = lazy_sort_index(idx_df.unstack())
        idx_df = lazy_sort_index(idx_df, axis=1)

        idx_map = {}
        for i, (_, row) in enumerate(idx_df.iterrows()):
            for j, real_idx in enumerate(row):
                if not np.isnan(real_idx):
                    idx_map[real_idx] = (i, j)
        return idx_df, idx_map

    @property
    def empty(self):
        return len(self) == 0

    def _get_indices(self, row: int, col: int) -> np.array:
        indices = self.idx_arr[max(row - self.step_len + 1, 0): row + 1, col]
        if len(indices) < self.step_len:
            indices = np.concatenate([np.full((self.step_len - len(indices),), np.nan), indices])

        if self.fillna_type == "ffill":
            indices = np_ffill(indices)
        elif self.fillna_type == "ffill+bfill":
            indices = np_ffill(np_ffill(indices)[::-1])[::-1]
        else:
            assert self.fillna_type == "none"
        return indices

    def _get_row_col(self, idx) -> Tuple[int]:
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + idx
            if self.start_idx <= real_idx < self.end_idx:
                i, j = self.idx_map[real_idx]
            else:
                raise KeyError(f"{real_idx} is out of [{self.start_idx}, {self.end_idx})")
        elif isinstance(idx, tuple):
            date, inst = idx
            date = pd.Timestamp(date)
            i = bisect.bisect_right(self.idx_df.index, date) - 1
            j = bisect.bisect_left(self.idx_df.columns, inst)
        else:
            raise NotImplementedError("索引类型不受支持。")
        return i, j

    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        mtit = (list, np.ndarray)
        if isinstance(idx, mtit):
            indices = [self._get_indices(*self._get_row_col(i)) for i in idx]
            indices = np.concatenate(indices)
        else:
            indices = self._get_indices(*self._get_row_col(idx))

        indices = np.nan_to_num(indices.astype(np.float64), nan=self.nan_idx).astype(int)
        data = self.data_arr[indices]
        if isinstance(idx, mtit):
            data = data.reshape(-1, self.step_len, *data.shape[1:])
        return data

    def __len__(self):
        return self.end_idx - self.start_idx


class DailyBatchSamplerRandom(Sampler):
    """
    简易的批次采样器示例，根据 time_id 做 daily batch，然后在日级别随机或不随机。
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # 按 time_id 分组后，统计每个 time_id 内的样本数量
        # 这里演示使用 data_source.get_index().groupby("time_id").size() 假设 index=[(time_id, stock_id)]
        # 如果你的 index 结构不同，请自行调整
        df_index = self.data_source.get_index()
        daily_count_series = pd.Series(index=df_index, dtype=pd.Float32Dtype)
        # 由于 df_index 其实是 MultiIndex，需要先获取 time_id
        time_ids = df_index.get_level_values("time_id")
        daily_count = pd.value_counts(time_ids).sort_index().values
        
        self.daily_count = daily_count
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                start_idx = self.daily_index[i]
                cnt = self.daily_count[i]
                yield np.arange(start_idx, start_idx + cnt)
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)
