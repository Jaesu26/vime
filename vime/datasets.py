from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.shape[0] == y.shape[0]
        assert X.ndim == y.ndim == 2
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = torch.FloatTensor(self.X[index])
        y = torch.FloatTensor(self.y[index])
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]


class UnlabeledDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        assert X.ndim == 2
        self.X = X

    def __getitem__(self, index: int) -> Tensor:
        x = torch.FloatTensor(self.X[index])
        return x

    def __len__(self) -> int:
        return self.X.shape[0]
