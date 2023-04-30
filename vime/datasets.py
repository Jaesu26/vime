from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.X[index]
        y = self.y[index]
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]


class UnlabeledDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        self.X = X

    def __getitem__(self, index: int) -> np.ndarray:
        x = self.X[index]
        return x

    def __len__(self) -> int:
        return self.X.shape[0]
