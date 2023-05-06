import torch
import torch.nn as nn
from torch import Tensor


class ConsistencyLoss(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(torch.var(x, dim=0))
