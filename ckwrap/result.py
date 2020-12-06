import numpy as np


class CkwrapResult(object):
    """Storage class for clustering results.

    Attributes:
        k: optimal number of clusters, as determined by BIC
        labels: array mapping cluster ids to values
        centers: cluster centers
        sizes: cluster sizes
        withinss: within-cluster sum of squares
        totss: total sum of squares
        BIC: Bayesian information criteria
        betweenss: sum of squares between cluster and sample

    """

    def __init__(
        self,
        k: int,
        labels: np.ndarray,
        centers: np.ndarray,
        sizes: np.ndarray,
        withinss: np.ndarray,
        totss: float,
        BIC: np.ndarray,
    ):
        self.k = k
        self.labels = np.array(labels)
        self.centers = np.array(centers)
        self.sizes = np.array(sizes)
        self.withinss = np.array(withinss)
        self.totss = totss
        self.BIC = BIC

    @property
    def betweenss(self) -> float:
        return self.totss - np.sum(self.withinss)
