from typing import Optional, Tuple, Union

import numpy as np
from sklearn.utils import check_random_state


def mask_generator(
    p_masking: float,
    size: Tuple[int, ...],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> np.ndarray:
    random_state = check_random_state(random_state)
    mask = random_state.binomial(n=1, p=p_masking, size=size)
    return mask


def pretext_generator(
    X: np.ndarray,
    mask: np.ndarray,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    random_state = check_random_state(random_state)
    X_bar = X.copy()
    np.apply_along_axis(func1d=random_state.shuffle, axis=0, arr=X_bar)
    X_tilde = X * (1 - mask) + X_bar * mask  # Corrupts samples(=X)
    corruption_mask = (X != X_tilde).astype(np.float32)
    return X_tilde, corruption_mask
