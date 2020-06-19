import numpy as np
from scipy.stats import dgamma

import ckwrap


def test_weighted_input():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([-1.0, 2.0, 4.0, 5.0, 6.0])
        y = np.array([4.0, 3.0, 1.0, 1.0, 1.0])
        result = ckwrap.ckmeans(x, 3, weights=y, method=method)
        assert np.all(result.labels + 1 == [1, 2, 3, 3, 3])
        assert np.allclose(result.centers, [-1.0, 2.0, 5.0])
        assert np.allclose(result.sizes, [4.0, 3.0, 3.0])
        assert np.allclose(result.withinss, [0.0, 0.0, 2.0])

        # Range of k values
        x = np.array([-0.9, 1.0, 1.1, 1.9, 2.0, 2.1])
        y = np.array([3.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        result = ckwrap.ckmeans(x, (1, 6), weights=y, method=method)
        assert np.allclose(
            result.centers, [-0.9, (1.0 + 2.2) / 3.0, (1.9 * 2.0 + 2.0 + 2.1) / 4.0]
        )
        assert np.allclose(result.sizes, [3.0, 3.0, 4.0])


def test_unweighted():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([-1.0, 2.0, -1.0, 2.0, 4.0, 5.0, 6.0, -1.0, 2.0, -1.0])
        result = ckwrap.ckmeans(x, 3, method=method)
        assert np.all(result.labels + 1 == [1, 2, 1, 2, 3, 3, 3, 1, 2, 1])
        assert np.allclose(result.centers, [-1.0, 2.0, 5.0])
        assert np.allclose(result.sizes, [4.0, 3.0, 3.0])
        assert np.allclose(result.withinss, [0.0, 0.0, 2.0])

        totss = np.sum((x - np.sum(x) / x.size) ** 2)
        assert np.allclose(result.totss, totss)
        assert np.allclose(np.sum(result.withinss), 2.0)
        assert np.allclose(np.sum(result.betweenss), totss - 2.0)


def test_n_less_or_equal_to_k():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([3.0, 2.0, -5.4, 0.1])
        result = ckwrap.ckmeans(x, 4, method=method)
        assert np.all(result.labels + 1 == [4, 3, 1, 2])
        assert np.allclose(result.centers, [-5.4, 0.1, 2.0, 3.0])
        assert np.allclose(result.sizes, [1.0, 1.0, 1.0, 1.0])
        assert np.allclose(result.withinss, [0.0, 0.0, 0.0, 0.0])


def test_k_equal_2():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.arange(1, 11)
        result = ckwrap.ckmeans(x, 2, method=method)
        assert np.all(result.labels + 1 == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        assert np.allclose(result.centers, [3.0, 8.0])
        assert np.allclose(result.sizes, [5.0, 5.0])
        assert np.allclose(result.withinss, [10.0, 10.0])


def test_k_equal_1():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([-2.5, -2.5, -2.5, -2.5])
        result = ckwrap.ckmeans(x, 1, method=method)
        assert np.all(result.labels + 1 == [1, 1, 1, 1])
        assert np.allclose(result.centers, [-2.5])
        assert np.allclose(result.sizes, [4.0])
        assert np.allclose(result.withinss, [0.0])

        x = np.arange(1, 101)
        result = ckwrap.ckmeans(x, 1, method=method)
        assert np.allclose(result.sizes, [100.0])


def test_n_equal_10_k_equal_3():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        result = ckwrap.ckmeans(x, 3, method=method)
        assert np.all(result.labels + 1 == [3, 3, 3, 3, 1, 1, 1, 2, 2, 2])
        assert np.allclose(result.centers, [1.0, 2.0, 3.0])
        assert np.allclose(result.sizes, [3.0, 3.0, 4.0])
        assert np.allclose(result.withinss, [0.0, 0.0, 0.0])


def test_n_equal_14_k_equal_8():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([-3, 2.2, -6, 7, 9, 11, -6.3, 75, 82.6, 32.3, -9.5, 62.5, 7, 95.2])
        result = ckwrap.ckmeans(x, 8, method=method)
        assert np.all(result.labels + 1 == [2, 2, 1, 3, 3, 3, 1, 6, 7, 4, 1, 5, 3, 8])
        assert np.allclose(
            result.centers, [-7.266666667, -0.4, 8.5, 32.3, 62.5, 75.0, 82.6, 95.2]
        )
        assert np.allclose(result.sizes, [3.0, 2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert np.allclose(
            result.withinss, [7.526666667, 13.52, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )


def test_estimate_k_set_1():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([0.9, 1.0, 1.1, 1.9, 2.0, 2.1])
        result = ckwrap.ckmeans(x, (1, 6), method=method)
        assert np.allclose(result.sizes, [3.0, 3.0])

        x = x[::-1]
        result = ckwrap.ckmeans(x, (1, 6), method=method)
        assert np.allclose(result.sizes, [3.0, 3.0])

        x = np.arange(1, 11)
        result = ckwrap.ckmeans(x, (1, 10), method=method)
        assert np.allclose(result.sizes, [10.0])


def test_estimate_k_set_2():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([3.5, 3.6, 3.7, 3.1, 1.1, 0.9, 0.8, 2.2, 1.9, 2.1])
        result = ckwrap.ckmeans(x, (2, 5), method=method)
        assert np.all(result.labels + 1 == [3, 3, 3, 3, 1, 1, 1, 2, 2, 2])
        assert np.allclose(result.centers, [0.933333333333, 2.066666666667, 3.475])
        assert np.allclose(result.sizes, [3.0, 3.0, 4.0])
        assert np.allclose(result.withinss, [0.0466666666667, 0.0466666666667, 0.2075])


def test_estimate_k_set_3_cosine():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.cos(np.arange(-10, 11))
        result = ckwrap.ckmeans(x, (1, 9), method=method)
        assert np.all(
            result.labels + 1
            == [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1]
        )
        assert np.allclose(result.centers, [-0.6592474631, 0.6751193405])
        assert np.allclose(result.sizes, [12.0, 9.0])
        assert np.allclose(result.withinss, [1.0564793100, 0.6232976959])


def test_estimate_k_set_4_gamma():
    for method in ["linear", "loglinear", "quadratic"]:
        x = (
            dgamma.pdf(np.arange(1, 10.5, 0.5), 2.0) * 2.0
        )  # times 2 for 1/2 in stats.dgamma
        result = ckwrap.ckmeans(x, (1, 9), method=method)
        assert np.all(
            result.labels + 1
            == [3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        assert np.allclose(
            result.centers, [0.01702193495, 0.15342151455, 0.32441508262]
        )
        assert np.allclose(result.sizes, [13, 3, 3])
        assert np.allclose(
            result.withinss, [0.006126754998, 0.004977009034, 0.004883305120]
        )
