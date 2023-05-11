from typing import List, Sequence, Tuple, Union

import numpy as np
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
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        cat_indices: Sequence[int] = (),
        cat_dims: Sequence[int] = (),
        cat_embedding_dim: Union[Sequence[int], int] = 2,
    ) -> None:
        super().__init__()
        self._encoder = Encoder(input_dim, hidden_dims, cat_indices, cat_dims, cat_embedding_dim)
        representation_dim = hidden_dims[-1] + self._encoder.embeddings.last_additional_dim
        self.feature_vector_estimator = nn.Linear(representation_dim, representation_dim)
        self.mask_vector_estimator = nn.Linear(representation_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._encoder(x)
        x_hat = self.feature_vector_estimator(z)
        mask_hat = self.mask_vector_estimator(z)
        return x_hat, mask_hat

    @property
    def encoder(self) -> nn.Module:
        return self._encoder


class Encoder(nn.Module, InitializerMixin):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        cat_indices: Sequence[int] = (),
        cat_dims: Sequence[int] = (),
        cat_embedding_dim: Union[Sequence[int], int] = 2,
    ) -> None:
        super().__init__()
        self.embeddings = EmbeddingGenerator(input_dim, cat_indices, cat_dims, cat_embedding_dim)
        input_dim = input_dim if self.embeddings.skip_embedding else self.embeddings.post_embedding_dim
        representation_dim = hidden_dims[-1] + self.embeddings.last_additional_dim
        in_dims = [input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims[:-1] + [representation_dim]
        self.encoder = nn.Sequential(*[get_block(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x_embedded = self.embeddings(x)
        z = self.encoder(x_embedded)
        return z


def get_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        PermuteBeforeBN1d(),  # For augmenting samples when running vime-semi
        nn.BatchNorm1d(out_dim),
        PermuteAfterBN1d(),  # For augmenting samples when running vime-semi
        nn.Mish(),
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


class EmbeddingGenerator(nn.Module):
    """Classical embedding generator.

    Args:
        input_dim: The number of features.
        cat_indices: The positional index for each categorical feature.
        cat_dims: The number of modalities for each categorical feature.
            If the list is empty, no embeddings will be done.
        cat_embedding_dim: The embedding dimension for each categorical feature.
            If int, the same embedding dimension will be used for all categorical features.

    References:
         https://github.com/dreamquark-ai/tabnet/blob/v4.0/pytorch_tabnet/tab_network.py#L778
    """

    def __init__(
        self,
        input_dim: int,
        cat_indices: Sequence[int] = (),
        cat_dims: Sequence[int] = (),
        cat_embedding_dim: Union[Sequence[int], int] = 2,
    ) -> None:
        super().__init__()
        self.cat_indices, self.cat_dims, self.cat_embedding_dims = self._verify_params(
            cat_indices, cat_dims, cat_embedding_dim
        )
        self.num_cats = len(self.cat_indices)
        self.total_cat_dim = sum(self.cat_dims)
        self.cont_indices = np.delete(range(input_dim), self.cat_indices)
        self.skip_embedding = True if self.num_cats else False
        self.post_embedding_dim = input_dim + sum(self.cat_embedding_dims) - self.num_cats
        self.last_additional_dim = self.total_cat_dim - self.num_cats
        self._sort_indices_for_seed()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_dim, cat_embedding_dim)
                for cat_dim, cat_embedding_dim in zip(self.cat_dims, self.cat_embedding_dims)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.skip_embedding:
            return x
        x_continuous = x[:, self.cont_indices]
        cats_embedded = [embedding(x[:, cat_index]) for cat_index, embedding in zip(self.cat_indices, self.embeddings)]
        x_embedded = torch.cat((x_continuous, *cats_embedded), dim=1)
        return x_embedded

    def _verify_params(
        self,
        cat_indices: Sequence[int],
        cat_dims: Sequence[int],
        cat_embedding_dim: int = Union[Sequence[int], int],
    ) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
        if bool(cat_indices) ^ bool(cat_dims):
            if cat_indices:
                message = "If cat_indices is non-empty, cat_dims must be defined as a list of same length."
            else:
                message = "If cat_dims is non-empty, cat_indices must be defined as a list of same length."
            raise ValueError(message)
        if len(cat_indices) != len(cat_dims):
            raise ValueError("cat_indices and cat_dims must have the same length.")
        if isinstance(cat_embedding_dim, int):
            cat_embedding_dims = [cat_embedding_dim] * len(cat_indices)
        else:
            cat_embedding_dims = cat_embedding_dim
        if len(cat_embedding_dims) != len(cat_dims):
            raise ValueError(
                "cat_embedding_dims and cat_dims must have the same length. "
                f"Got: {len(cat_embedding_dims)} and {len(cat_dims)}"
            )
        return cat_indices, cat_dims, cat_embedding_dims

    def _sort_indices_for_seed(self) -> None:
        if not self.num_cats:
            return
        sorted_indices = np.argsort(self.cat_indices)
        self.cat_indices = sorted(self.cat_indices)
        self.cat_dims = [self.cat_dims[i] for i in sorted_indices]
        self.cat_embedding_dims = [self.cat_embedding_dims[i] for i in sorted_indices]


class VIMESemiNetwork(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.predictor = Predictor(input_dim, hidden_dims, num_classes)
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
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int) -> None:
        super().__init__()
        in_dims = [input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.fc = nn.Sequential(*[get_block(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])
        self.classifier = nn.Linear(out_dims[-1], num_classes)
        self._init_weights()

    def forward(self, z: Tensor) -> Tensor:
        z = self.fc(z)
        y_hat = self.classifier(z)
        return y_hat


class MLP(nn.Module, InitializerMixin):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = get_block(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        z = self.mlp(x)
        y_hat = self.fc(z)
        return y_hat
