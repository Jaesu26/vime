import torch
import torch.nn as nn
from torch import Tensor


class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.log(input)
        input = torch.clamp(input, -100)  # Improve numerical stability.
        loss = self.nll_loss(input, target)
        return loss


class ConsistencyLoss(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(torch.var(x, dim=0))
