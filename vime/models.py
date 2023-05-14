from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class WeightsInitializerMixin:
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
        cat_indices: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_embedding_dim: Union[int, List[int]] = 2,
    ) -> None:
        super().__init__()
        self._encoder = Encoder(input_dim, hidden_dims, cat_indices, cat_dims, cat_embedding_dim)
        representation_dim = hidden_dims[-1]
        total_dim_after_ohe = self._encoder.embedder.total_dim_after_ohe
        cat_dims = self._encoder.embedder.cat_dims
        self.feature_vector_estimator = FeatureVectorEstimator(representation_dim, total_dim_after_ohe, cat_dims)
        self.mask_vector_estimator = MaskVectorEstimator(representation_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self._encoder(x)
        x_hat = self.feature_vector_estimator(z)
        mask_hat = self.mask_vector_estimator(z)
        return x_hat, mask_hat

    @property
    def encoder(self) -> nn.Module:
        return self._encoder


class Encoder(nn.Module, WeightsInitializerMixin):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        cat_indices: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_embedding_dim: Union[int, List[int]] = 2,
    ) -> None:
        super().__init__()
        self.embedder = EmbeddingGenerator(input_dim, cat_indices, cat_dims, cat_embedding_dim)
        input_dim = self.embedder.post_embedding_dim
        in_dims = [input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.encoder = nn.Sequential(*[get_block(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x_embedded = self.embedder(x)
        z = self.encoder(x_embedded)
        return z


class MaskVectorEstimator(nn.Module):
    def __init__(self, representation_dim: int, mask_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(representation_dim, mask_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        mask_logit = self.fc(x)
        mask_hat = self.sigmoid(mask_logit)
        return mask_hat


class FeatureVectorEstimator(nn.Module):
    def __init__(self, representation_dim: int, total_dim_after_ohe: int, cat_dims: List[int]) -> None:
        super().__init__()
        self.fc = nn.Linear(representation_dim, total_dim_after_ohe)
        self.softmax = nn.Softmax(dim=1)
        self.cat_dims = [0] + cat_dims
        self.start_indices = np.cumsum(self.cat_dims)[:-1]
        self.end_indices = np.cumsum(self.cat_dims)[1:]

    def forward(self, x: Tensor) -> Tensor:
        x_hat = self.fc(x)
        for start_index, end_index in zip(self.start_indices, self.end_indices):
            x_hat[:, start_index:end_index] = self.softmax(x_hat[:, start_index:end_index])
        return x_hat


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
        cat_dims: The number of unique values for each categorical features.
            If the list is empty, no embeddings will be done.
        cat_embedding_dim: The embedding dimension for each categorical feature.
            If int, the same embedding dimension will be used for all categorical features.

    References:
         https://github.com/dreamquark-ai/tabnet/blob/v4.0/pytorch_tabnet/tab_network.py#L778
    """

    def __init__(
        self,
        input_dim: int,
        cat_indices: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_embedding_dim: Union[int, List[int]] = 2,
    ) -> None:
        super().__init__()
        cat_indices = cat_indices or []
        cat_dims = cat_dims or []
        self.cat_indices, self.cat_dims, self.cat_embedding_dims = self._check_embedding_params(
            cat_indices, cat_dims, cat_embedding_dim
        )
        self.cont_indices = np.delete(range(input_dim), self.cat_indices)
        self.total_cat_dim = sum(self.cat_dims)
        self.num_categories = len(self.cat_indices)
        if self.num_categories:
            self.skip_embedding = True
            self.post_embedding_dim = input_dim
            return
        self.skip_embedding = False
        self.post_embedding_dim = input_dim + sum(self.cat_embedding_dims) - self.num_categories
        self.total_dim_after_ohe = input_dim + self.total_cat_dim - self.num_categories
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_dim, cat_embedding_dim)
                for cat_dim, cat_embedding_dim in zip(self.cat_dims, self.cat_embedding_dims)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.skip_embedding:
            return x
        xs_categorical_embedded = [emb(x[:, cat_index]) for cat_index, emb in zip(self.cat_indices, self.embeddings)]
        x_continuous = x[:, self.cont_indices]
        x_embedded = torch.cat((*xs_categorical_embedded, x_continuous), dim=1)
        return x_embedded

    def _check_embedding_params(
        self,
        cat_indices: List[int],
        cat_dims: List[int],
        cat_embedding_dim: int = Union[int, List[int]],
    ) -> Tuple[List[int], List[int], List[int]]:
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
        # Rearrange to get reproducible seeds with different ordering
        if not cat_indices:
            sorted_indices = np.argsort(cat_indices)
            cat_indices.sort()
            cat_dims = [cat_dims[index] for index in sorted_indices]
            cat_embedding_dims = [cat_embedding_dims[index] for index in sorted_indices]
        return cat_indices, cat_dims, cat_embedding_dims


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


class Predictor(nn.Module, WeightsInitializerMixin):
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


class MLP(nn.Module, WeightsInitializerMixin):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = get_block(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        z = self.mlp(x)
        y_hat = self.fc(z)
        return y_hat
