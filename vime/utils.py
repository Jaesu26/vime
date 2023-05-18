from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def check_y_dtype(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    num_classes = len(classes)
    is_classification = np.array_equal(classes, np.arange(num_classes))
    is_multiclass = is_classification and num_classes >= 3
    if is_multiclass:
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float32)
    return y


def check_rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.default_rng function.")


def mask_generator(
    p_masking: float,
    size: Tuple[int, ...],
    generator: np.random.Generator,
) -> Tensor:
    mask = generator.binomial(n=1, p=p_masking, size=size)
    mask = torch.from_numpy(mask)
    return mask


def pretext_generator(
    x: Tensor,
    mask: Tensor,
    generator: np.random.Generator,
) -> Tuple[Tensor, Tensor]:
    x_bar = generator.permuted(x.numpy(), axis=0)
    x_bar = torch.from_numpy(x_bar)
    x_tilde = x * (1 - mask) + x_bar * mask  # Corrupts samples(=x)
    corruption_mask = x.ne(x_tilde).float()
    return x_tilde, corruption_mask
