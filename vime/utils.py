import numpy as np


def mask_generator(p_m, x):
    """Generates mask vector.

    Args:
      p_m: Corruption probability.
      x: Feature matrix.

    Returns:
      mask: Binary mask matrix.
    """
    mask = np.random.binomial(n=1, p=p_m, size=x.shape)
    return mask


def pretext_generator(mask, x):
    """Generates corrupted samples.

    Args:
      mask: Mask matrix.
      x: Feature matrix.

    Returns:
      mask_new: Final mask matrix after corruption.
      x_tilde: Corrupted feature matrix.
    """
    x_bar = x.copy()
    np.apply_along_axis(func1d=np.random.shuffle, axis=0, arr=x_bar)
    x_tilde = x * (1 - mask) + x_bar * mask     # Corrupt samples
    mask_new = (x != x_tilde).astype(np.float32)  # Define new mask matrix
    return mask_new, x_tilde
