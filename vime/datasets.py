from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.X[index]
        y = self.y[index]
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]


class UnlabeledDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tensor:
        x = self.X[index]
        return x

    def __len__(self) -> int:
        return self.X.shape[0]
