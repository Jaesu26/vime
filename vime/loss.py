import torch
import torch.nn as nn
from torch import Tensor


class CELoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.nll_loss = nn.NLLLoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.clamp_(input, self.eps, 1 - self.eps)
        input = torch.log(input)
        loss = self.nll_loss(input, target)
        return loss


class ConsistencyLoss(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(torch.var(x, dim=0))
