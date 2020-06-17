import numpy as np

import _ckwrap
from ckwrap.result import CkwrapResult

from typing import Union, Tuple

means_criteria = 0
medians_criteria = 1
segs_criteria = 2


def _ckcluster(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]],
    y: np.ndarray,
    method: str,
    criteria: int,
) -> CkwrapResult:
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

    if y is None:
        y = np.array([1.0], dtype=np.float64)
    elif y.ndim > 1:
        raise ValueError("'y' must be 1-dimensional.")

    if criteria == 2 and x.size != y.size:
        raise ValueError("Segs requires 'x' and 'y' to be same size.")

    if method not in ["linear", "loglinear", "quadratic"]:
        raise ValueError("Method must be one of 'linear', 'loglinear', 'quadratic'.")

    return _ckwrap.ckcluster(x, y, k[0], k[1], method, criteria)


def ckmeans(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    weights: np.ndarray = None,
    method: str = "linear",
) -> CkwrapResult:
    return _ckcluster(x, k, weights, method, means_criteria)


def ckmedians(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    weights: np.ndarray = None,
    method: str = "linear",
) -> CkwrapResult:
    return _ckcluster(x, k, weights, method, medians_criteria)


def cksegs(
    x: np.ndarray,
    y: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    method: str = "linear",
) -> CkwrapResult:
    return _ckcluster(x, k, y, method, segs_criteria)
