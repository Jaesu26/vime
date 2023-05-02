from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import Tensor


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
    X: Tensor,
    mask: Tensor,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[Tensor, Tensor]:
    random_state = check_random_state(random_state)
    X_bar = X.numpy().copy()
    np.apply_along_axis(func1d=random_state.shuffle, axis=0, arr=X_bar)
    X_bar = torch.from_numpy(X_bar)
    X_tilde = X * (1 - mask) + X_bar * mask  # Corrupts samples(=X)
    corruption_mask = X.ne(X_tilde).float()
    return X_tilde, corruption_mask
