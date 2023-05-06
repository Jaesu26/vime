from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class InitializerMixin:
    def _init_weights(self: nn.Module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class VIMESelfNetwork(nn.Module):
    def __init__(self, in_features_list: List[int], out_features_list: List[int]) -> None:
        super().__init__()
        dim = in_features_list[0]
        representation_dim = out_features_list[-1]
        self._encoder = Encoder(in_features_list, out_features_list)
        self.feature_vector_estimator = nn.Linear(representation_dim, dim)
        self.mask_vector_estimator = nn.Linear(representation_dim, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._encoder(x)
        x_hat = self.feature_vector_estimator(z)
        mask_hat = self.mask_vector_estimator(z)
        return x_hat, mask_hat

    @property
    def encoder(self) -> nn.Module:
        return self._encoder


class Encoder(nn.Module, InitializerMixin):
    def __init__(self, in_features_list: List[int], out_features_list: List[int]) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                get_block(in_features, out_features)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return z


def get_block(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        PermuteBeforeBN1d(),  # For augmenting samples when running vime-semi
        nn.BatchNorm1d(out_features),
        PermuteAfterBN1d(),  # For augmenting samples when running vime-semi
        nn.GELU(),
    )


class PermuteBeforeBN1d(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            return x
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # Shape of x: (K, B, C) -> (B, C, K)
        return x.permute(1, 2, 0)


class PermuteAfterBN1d(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            return x
        # Shape of x: (B, C, K) -> (K, B, C)
        return x.permute(2, 0, 1)


class VIMESemiNetwork(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        in_features_list: List[int],
        out_features_list: List[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.predictor = Predictor(in_features_list, out_features_list, num_classes)
        self.freeze_encoder()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.pretrained_encoder(x)
        y_hat = self.predictor(z)
        return y_hat

    def freeze_encoder(self) -> None:
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        self.pretrained_encoder.eval()


class Predictor(nn.Module, InitializerMixin):
    def __init__(self, in_features_list: List[int], out_features_list: List[int], num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            *[
                get_block(in_features, out_features)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )
        self.classifier = nn.Linear(out_features_list[-1], num_classes)
        self._init_weights()

    def forward(self, z: Tensor) -> Tensor:
        z = self.fc(z)
        y_hat = self.classifier(z)
        return y_hat


class MLP(nn.Module, InitializerMixin):
    def __init__(self, dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = get_block(dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        z = self.mlp(x)
        y_hat = self.fc(z)
        return y_hat
