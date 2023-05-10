from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConsistencyLoss(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(torch.var(x, dim=0))


class ReconstructionLoss(nn.Module):
    def forward(self, x_continuous: Tensor, *x_categorical: Tuple[Tensor, ...]) -> Tensor:
        return x_continuous
