from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.utils import check_random_state
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


def mask_generator(
    p_masking: float,
    size: Tuple[int, ...],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tensor:
    random_state = check_random_state(random_state)
    mask = random_state.binomial(n=1, p=p_masking, size=size)
    mask = torch.from_numpy(mask)
    return mask


def pretext_generator(
    x: Tensor,
    mask: Tensor,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Tensor, Tensor]:
    random_state = check_random_state(random_state)
    x_bar = x.numpy().copy()
    np.apply_along_axis(func1d=random_state.shuffle, axis=0, arr=x_bar)
    x_bar = torch.from_numpy(x_bar)
    x_tilde = x * (1 - mask) + x_bar * mask  # Corrupts samples(=x)
    corruption_mask = x.ne(x_tilde).float()
    return x_tilde, corruption_mask
