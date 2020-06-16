#cython: language_level=3
#distutils: language=c++

from libcpp.string cimport string


cdef extern from "../Ckmeans.1d.dp/src/Ckmeans.1d.dp.cpp":
    cdef extern enum DISSIMILARITY:
        L1,
        L2,
        L2Y

    void kmeans_1d_dp(double *x, int N, double *y,
                      int Kmin, int Kmax,
                      int* cluster, double *centers,
                      double *withinss, double *size,
                      double *BIC,
                      string& estimate_k,
                      string& method,
                      DISSIMILARITY criterion)
