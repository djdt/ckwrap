import numpy as np


class CkwrapResult(object):
    def __init__(
        self,
        k: int,
        labels: np.ndarray,
        centers: np.ndarray,
        sizes: np.ndarray,
        wintinss: np.ndarray,
        totss: float,
        BIC: np.ndarray,
    ):
        self.k = k
        self.labels = np.array(labels)
        self.centers = np.array(centers)
        self.sizes = np.array(sizes)
        self.wintinss = np.array(wintinss)
        self.totss = totss
        self.BIC = BIC

    @property
    def betweenss(self) -> float:
        return self.totss - np.sum(self.wintinss)
