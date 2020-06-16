import numpy as np

import _ckwrap

from typing import Union, Tuple

means_criteria = 0
medians_crtieria = 1


def _ckcluster(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]],
    weights: np.ndarray,
    method: str,
    criteria: int,
) -> dict:
    if x.ndim > 1:
        raise ValueError("'x' must be 1-dimensional.")

    if isinstance(k, int):
        k = k, k
    if k[0] < 1:
        raise ValueError("Minimum 'k' must be greater than 1.")
    if k[1] > np.unique(x).size:
        raise ValueError(
            "Max 'k' must be smaller than the number of unique 'x' values."
        )

    if weights is None:
        weights = np.array([1.0], dtype=np.float64)

    if method not in ["linear", "loglinear", "quadratic"]:
        raise ValueError("Method must be one of 'linear', 'loglinear', 'quadratic'.")

    return _ckwrap.ckcluster(x, weights, k[0], k[1], method, criteria)


def ckmeans(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    weights: np.ndarray = None,
    method: str = "linear",
) -> dict:
    return _ckcluster(x, k, weights, method, medians_crtieria)


def ckmedians(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    weights: np.ndarray = None,
    method: str = "linear",
) -> dict:
    return _ckcluster(x, k, weights, method, medians_crtieria)
