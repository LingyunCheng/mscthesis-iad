from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from . import Base


class IADE(Base):
    def __init__(self, Z, X, lr=0.1, iterations=2000, plot=False, eps=0.4):
        super().__init__(Z, X, lr, iterations, plot)
        self.X = X
        self.Z = Z
        self.eps = eps

        self.clf_rf = RandomForestClassifier(n_estimators=100)
        self.clf_lg = LogisticRegression(C=0.01, solver="lbfgs", penalty="l2") # RandomForestClassifier(n_estimators=100)
        self.scores_original = np.sum(self.Z, axis=1).reshape(-1)

        self.scores = self.get_scores()

    # =====================================
    def get_scores_clf(self, D, clf):
        if len(self.H[1]) == 0 or len(self.H[-1]) == 0:
            return self.scores_original

        ids_labeled, labels = self.get_labeled_ids()
        DL = D[ids_labeled]
        clf.fit(DL, labels)

        i = clf.classes_.reshape(-1).tolist().index(1)
        probas = clf.predict_proba(D)[:, i].reshape(-1)

        return probas

    # =====================================
    def get_scores(self):
        scores_clf_X = self.get_scores_clf(self.X, self.clf_rf)
        scores_clf_Z = self.get_scores_clf(self.Z, self.clf_lg)

        if np.random.uniform(0, 1) > self.eps:
            return scores_clf_X
        else:
            return scores_clf_Z
            #return self.scores_original

    # =====================================
    def update(self, u, yu):
        super().update(u, yu)
        self.scores = self.get_scores()
