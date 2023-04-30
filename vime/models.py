from copy import deepcopy
from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


def get_block(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.GELU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_features_list: List[int], out_features_list: List[int]) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                get_block(in_features, out_features)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return z


class VIMESelfModule(nn.Module):
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
        return deepcopy(self._encoder)


class Predictor(nn.Module):
    def __init__(self, in_features_list: List[int], out_features_list: List[int], num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *[
                get_block(in_features, out_features)
                for in_features, out_features in zip(in_features_list, out_features_list)
            ]
        )
        self.classifier = nn.Linear(out_features_list[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.classifier(x)
        return x


class VIMESemiModule(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        y_hat = self.predictor(z)
        return y_hat
