import numpy as np

import _ckwrap
from ckwrap.result import CkwrapResult

from typing import Union, Tuple

L1_criteria = 0
L2_criteria = 1
L2Y_criteria = 2


def _ckcluster(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]],
    y: np.ndarray,
    method: str,
    criteria: int,
) -> CkwrapResult:
    x = np.ascontiguousarray(x, dtype=np.float64)
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
    else:
        y = np.ascontiguousarray(y, dtype=np.float64)

    if y.ndim > 1:
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
    """Returns the optimal k-means clustering result of `x` for k in range `k`.

    Uses ckmeans.1d.dp with `L2` criteria to minimize within-cluster sums of squared
    distances.
    Bayesian information criterion selects optimal `k` when `k` is passed as a range.
    Optimal weighted clustering can be performed by passing in `weights`, the
    default is equally weighted clustering.
    All `method`s return the same result and only differ in speed. From
    fastest to slowest the options are 'linear', 'loglinear' and 'quadratic'.

    Args:
        x: array, one dimensional
        k: single or range of k to test
        weights: weights for clustering, same shape as x
        method: options {'linear', 'loglinear', 'quadratic'}

    Returns:
        optimal clustering result
    """
    return _ckcluster(x, k, weights, method, L2_criteria)


def ckmedians(
    x: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    method: str = "linear",
) -> CkwrapResult:
    """Returns the optimal k-medians clustering result of `x` for k in range `k`.

    Uses ckmeans.1d.dp with `L1` criteria to minimize within-clusters sum fo distances.
    Bayesian information criterion selects optimal `k` when `k` is passed as a range.
    All `method`s return the same result and only differ in speed. From
    fastest to slowest the options are 'linear', 'loglinear' and 'quadratic'.

    Args:
        x: array, one dimensional
        k: single or range of k to test
        method: options {'linear', 'loglinear', 'quadratic'}

    Returns:
        optimal clustering result
    """
    return _ckcluster(x, k, None, method, L1_criteria)


def cksegs(
    y: np.ndarray,
    k: Union[int, Tuple[int, int]] = (1, 9),
    x: np.ndarray = None,
    method: str = "quadratic",
) -> CkwrapResult:
    """Minimizes within-cluster sum of squared distance on `y`.

    Offers optimal piecewise constant approximation of `y` within clusters of `x`.
    Only `method` 'quadratic' guarantees optimality.
    Bayesian information criterion selects optimal `k` when `k` is passed as a range.

    Args:
        x: array, one dimensional
        k: single or range of k to test
        x: array, same shape as x, default is range(1, len(y) + 1)
        method: options {'quadratic', 'linear', 'loglinear'}

    Returns:
        optimal clustering result
    """
    if x is None:
        x = np.arange(1, y.size + 1)
    return _ckcluster(x, k, y, method, L2Y_criteria)
