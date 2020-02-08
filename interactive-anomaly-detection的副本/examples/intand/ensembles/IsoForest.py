import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import IsolationForest

class IsoForest:
    def __init__(self, n_trees=50, sparse=False, rs=None):
        self.n_trees = n_trees
        self.rs = rs if rs is None else rs * 1000
        self.transform = self.transform_sparse if sparse else self.transform_non_sparse

    def fit(self, X):
        if self.rs is None:
            self.ensemble = [IsolationForest(n_estimators=1, behaviour="new", contamination="auto").fit(X) for _ in range(self.n_trees)]
        else:
            self.ensemble = [IsolationForest(n_estimators=1, behaviour="new", contamination="auto", random_state=self.rs+i).fit(X) for i in
                             range(self.n_trees)]
        return self

    def transform_sparse(self, X):
        S = [[] for _ in range(len(X))]

        for irf in self.ensemble:
            tree = irf.estimators_[0]

            leaves = tree.apply(X)
            unique_leaves = sorted(set(leaves))
            dic = {l: i for i, l in enumerate(unique_leaves)}
            leaves = [dic[l] for l in leaves]

            scores = -irf.score_samples(X)
            for i, x in enumerate(X):
                vec = [0 for _ in unique_leaves]
                vec[leaves[i]] = scores[i]
                S[i] += vec
        return np.array(S)

    def transform_non_sparse(self, X):
        S = [[-irf.score_samples([x])[0] for irf in self.ensemble] for x in X]
        return np.array(S)

