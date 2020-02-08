import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import IsolationForest


class Loda:
    def __init__(self, projections=50, bins=10):
        self.k = projections
        self.bins = bins
        self.rprog, self.histograms = None, None

    @staticmethod
    def get_bin_density(v, histogram):
        hist, bin_edges = histogram
        for i, be in enumerate(bin_edges):
            if v <= be: break
        i = max(i - 1, 0)
        return i, hist[i]

    def fit(self, X):
        if self.k is not None: self.rprog = GaussianRandomProjection(n_components=self.k).fit(X)
        XX = self.rprog.transform(X) if self.rprog is not None else X
        self.histograms = [np.histogram(XX[:, j], bins=self.bins, density=True) for j in range(XX.shape[1])]
        return self
    
    def transform(self, X):
        XX = self.rprog.transform(X) if self.rprog is not None else X
        anomaly_vect = lambda xx: [-np.log(self.get_bin_density(xx_j, histo)[1]) for (xx_j, histo) in zip(xx, self.histograms)]
        return np.array([anomaly_vect(xx) for xx in XX])
