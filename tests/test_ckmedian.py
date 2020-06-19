import numpy as np

import ckwrap


def test_unweighted():
    for method in ["linear", "loglinear", "quadratic"]:
        x = np.array([-1.0, 2.0, 4.0, 5.0, 6.0])
        result = ckwrap.ckmedians(x, 3, method=method)
        assert np.all(result.labels + 1 == [1, 2, 3, 3, 3])
        assert np.allclose(result.centers, [-1.0, 2.0, 5.0])
        assert np.allclose(result.sizes, [1.0, 1.0, 3.0])
        assert np.allclose(result.withinss, [0.0, 0.0, 2.0])

        x = np.array([-0.9, 1.0, 1.1, 1.9, 2.0, 2.05])
        result = ckwrap.ckmedians(x, (1, 6), method=method)
        assert np.allclose(result.centers, [-0.9, 1.0, 2.0])
        assert np.allclose(result.sizes, [1.0, 2.0, 3.0])
