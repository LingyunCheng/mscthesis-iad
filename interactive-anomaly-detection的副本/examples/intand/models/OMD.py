import numpy as np
from . import Base


class OMD(Base):
    def __init__(self, Z, X=None, lr=0.1, iterations=2000, plot=False, non_negative_w=True, loss="linear"):
        super().__init__(Z, X, lr, iterations, plot)
        self.non_negative_w = non_negative_w
        self.loss = self.linear_loss if loss == "linear" else self.log_likelihood_loss
        self.gradient_loss = self.gradient_linear_loss if loss == "linear" else self.gradient_log_likelihood_loss
        self.w = np.ones(Z.shape[1])
        self.scores = self.get_scores(self.Z)

    def get_scores(self, Z):
        return Z @ self.w

    def linear_loss(self, u, yu):
        return -yu * (self.w @ self.Z[u])

    def gradient_linear_loss(self, u, yu):
        return -yu * self.Z[u]

    def log_likelihood_loss(self, u, yu):
        exp_s = np.exp(self.Z @ self.w)
        pu = exp_s[u] / np.sum(exp_s)
        return -yu * np.log(pu)

    def gradient_log_likelihood_loss(self, u, yu):
        exp_s = np.exp(self.Z @ self.w)
        pu = exp_s[u] / np.sum(exp_s)
        dp_dw = pu * (self.Z[u] - np.sum([self.Z[i] * e for i, e in enumerate(exp_s)], axis=0) / np.sum(exp_s))
        return -yu * (1 / pu) * dp_dw

    def update(self, u, yu):
        super().update(u, yu)

        self.w = self.w - self.lr * self.gradient_loss(u, yu)
        if self.non_negative_w:
            self.w[self.w < 0] = 0.1

        self.scores = self.get_scores(self.Z)
