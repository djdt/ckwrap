#cython: language_level=3
#distutils: language=c++

from libcpp.string cimport string
from ckwrap cimport kmeans_1d_dp

import numpy as np


def ckcluster(double[:] x, double[:] y, int min_k, int max_k, str method, int criterion) -> dict:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    cdef int nx = len(x)
    cdef int ny = len(y)

    cdef int [:] clusters = np.empty(x.size, dtype=np.int32)
    cdef double [:] centers = np.empty(max_k, dtype=np.float64)
    cdef double [:] withinss = np.zeros(max_k, dtype=np.float64)
    cdef double [:] sizes = np.zeros(max_k, dtype=np.float64)
    cdef double [:] BICs = np.empty(max_k - min_k + 1, dtype=np.float64)

    cdef string cpp_method = method.encode("UTF-8")

    kmeans_1d_dp.kmeans_1d_dp(
            &x[0], nx,
            &y[0] if nx == ny else NULL,
            min_k, max_k,
            &clusters[0],
            &centers[0],
            &withinss[0],
            &sizes[0],
            &BICs[0],
            string(b"BIC"),
            cpp_method,
            <kmeans_1d_dp.DISSIMILARITY>criterion,
    )

    cdef int k = np.count_nonzero(sizes)

    return {"k": k, "cluster": clusters, "centers": centers[:k], "withinss": withinss[:k], "size": sizes[:k]}
