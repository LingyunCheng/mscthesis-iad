import numpy as np
from . import Base


class AAD(Base):
    def __init__(self, Z, X=None, lr=0.1, iterations=2000, plot=False, tau=0.3, lmd=0.01, Ca=1, Cn=1, Cx=1):
        super().__init__(Z, X, lr, iterations, plot)
        self.tau = tau
        self.lmd = lmd
        self.Ca = Ca
        self.Cn = Cn
        self.Cx = Cx
        self.wp = np.ones(Z.shape[1])  # / (Z.shape[1] ** 0.5)
        self.w = np.ones(Z.shape[1])
        self.scores = self.get_scores(self.Z)

    # =====================================
    def get_q_tau(self):
        ids = np.argsort(-self.scores)
        i_tau = ids[int(self.tau * len(ids) - 1)]
        q_tau = self.scores[i_tau]
        z_tau = self.Z[i_tau]
        return q_tau, z_tau

    # =====================================
    def get_scores(self, Z):
        return Z @ self.w

    # =====================================
    def loss(self, ZL, yL, q_tau, z_tau):
        scores = self.get_scores(ZL)

        scores_pos, scores_neg = scores[yL == 1], scores[yL == -1]
        cost_1_pos = np.mean(np.maximum(0, (q_tau - scores_pos))) if len(scores_pos) > 0 else 0
        cost_1_neg = np.mean(np.maximum(0, -1 * (q_tau - scores_neg))) if len(scores_neg) > 0 else 0

        score_z_tau = self.get_scores(z_tau)
        cost_2 = np.mean(np.maximum(0, yL * (score_z_tau - scores)))

        cost_prior = 0.5 * np.linalg.norm(self.w - self.wp) ** 2
        return self.Ca * cost_1_pos + self.Cn * cost_1_neg + self.Cx * cost_2 + self.lmd * cost_prior

    # =====================================
    def gradient_loss(self, ZL, yL, q_tau, z_tau):
        def gradient_loss_i(zi, yi, with_tau=False):
            si = self.get_scores(zi)
            if (yi == 1 and si >= q_tau) or (yi == -1 and si < q_tau): return 0
            elif yi == 1 and si < q_tau: return -zi if (not with_tau) else (z_tau - zi)
            elif yi == -1 and si >= q_tau: return zi if (not with_tau) else (zi - z_tau)

        ZL_pos, ZL_neg = ZL[yL == 1], ZL[yL == -1]
        grad_a = np.mean([gradient_loss_i(zi, 1) for zi in ZL_pos], axis=0) if len(ZL_pos) > 0 else 0
        grad_n = np.mean([gradient_loss_i(zi, -1) for zi in ZL_neg], axis=0) if len(ZL_neg) > 0 else 0
        grad_tau = np.mean([gradient_loss_i(zi, yi, True) for (zi, yi) in zip(ZL, yL)], axis=0)
        grad_prior = self.w - self.wp

        grad = np.sum([self.Ca * grad_a, self.Cn * grad_n, self.Cx * grad_tau, self.lmd * grad_prior], axis=0)

        return grad

    # =====================================
    def minimize_w(self, ZL, yL, q_tau, z_tau): # We use the ADAM optimizer (i.e. gradient descent with momentum + rmsprop)
        avg_grad1 = 0; avg_grad2 = 0
        beta1 = 0.9; beta2 = 0.999; eps = 1e-07
        for itr in range(self.iterations):
            avg_grad1 = beta1 * avg_grad1 + (1 - beta1) * self.gradient_loss(ZL, yL, q_tau, z_tau)
            avg_grad2 = (beta2 * avg_grad2 + (1 - beta2) * (self.gradient_loss(ZL, yL, q_tau, z_tau) ** 2))
            avg_grad1_corr = avg_grad1 / (1 - beta1 ** (itr + 1))
            avg_grad2_corr = avg_grad2 / (1 - beta2 ** (itr + 1))
            self.w = self.w - self.lr * (avg_grad1_corr / (np.sqrt(avg_grad2_corr) + eps))

    # =====================================
    def update(self, u, yu):
        super().update(u, yu)

        ids_labeled, yL = self.get_labeled_ids()
        ZL = self.Z[ids_labeled]
        q_tau, z_tau = self.get_q_tau()

        self.minimize_w(ZL, yL, q_tau, z_tau)

        self.scores = self.get_scores(self.Z)
