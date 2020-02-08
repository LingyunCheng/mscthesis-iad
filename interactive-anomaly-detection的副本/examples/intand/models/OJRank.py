import numpy as np
from . import Base


def sigmoid(a):
    return 1 / (1 + np.exp(-(a)))


def sample(scores, size, inv=False, c=-0.99):
    arr = np.arange(len(scores))
    probs = 1. / np.array(scores) if inv else np.array(scores)

    ids = np.argsort(probs)[:len(probs) // 2]
    probs[ids] = 0

    probs = probs / np.sum(probs)
    if not inv: probs = (c * probs + 1) ** (1 / c)
    probs[np.isnan(probs)] = 0 # quick hack
    probs_saved = probs ############# for debug

    probs = probs / np.sum(probs)
    probs[probs < 0] = 0  # quick hack to avoid some decimal precision problems
    probs = probs / np.sum(probs)

    try: choice = np.random.choice(arr, size=size, replace=False, p=probs)
    except:
        try:
            choice = np.random.choice(arr, size=size, replace=True, p=probs)
        except:
            print("scores", scores)
            print("probs", probs)
            print("probs_saved", probs_saved)
            print("np.sum(probs_saved)", np.sum(probs_saved))

    return choice


def generate_pairs(u, yu, scores, H, k, delta):
    PH, PS = [], []
    a, z = np.argmax(scores), np.argmin(scores)

    if yu == 1:
        for v in H[-1]:
            p_uv = sigmoid(scores[a] - scores[z])
            PH.append([u, v, p_uv])

        size = np.clip(k - len(H[-1]), 0, None)
        for v in sample(scores, size, inv=True):
            p_uv = (1 + delta) * sigmoid(scores[u] - scores[v])
            p_uv = np.clip(p_uv, 0, 1)
            PS.append([u, v, p_uv])

    else:
        for v in H[1]:
            p_uv = 1 - sigmoid(scores[a] - scores[z])
            PH.append([u, v, p_uv])

        size = np.clip(k - len(H[1]), 0, None)
        for v in sample(scores, size, inv=False):
            p_uv = (1 - delta) * sigmoid(scores[u] - scores[v])
            p_uv = np.clip(p_uv, 0, 1)
            PS.append([u, v, p_uv])

    return np.array(PH + PS)


class OJRank(Base):
    def __init__(self, Z, X=None, lr=0.1, iterations=2000, plot=False, delta=0.1, k=20):
        super().__init__(Z, X, lr, iterations, plot)
        self.delta = delta
        self.k = k
        self.w = np.ones(Z.shape[1])
        self.scores = self.get_scores(self.Z)

    # =====================================
    def get_scores(self, Z):
        return Z @ self.w

    # =====================================
    def loss(self, pairs):
        U, V, P = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
        scores_U = self.get_scores(self.Z[U])
        scores_V = self.get_scores(self.Z[V])
        P_hat = sigmoid(scores_U - scores_V)
        P_hat = np.clip(P_hat, 0.000001, 0.999999)
        cost = np.sum(-P * np.log(P_hat) - (1 - P) * np.log(1 - P_hat))
        return cost

    # =====================================
    def gradient_loss(self, pairs):
        U, V, P = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
        scores_U = self.get_scores(self.Z[U])
        scores_V = self.get_scores(self.Z[V])
        res = (sigmoid(scores_U - scores_V) - P).reshape(-1, 1) * (self.Z[U] - self.Z[V])
        return np.sum(res, axis=0)

    # =====================================
    def minimize_w(self, pairs): # We use the ADAM optimizer (i.e. gradient descent with momentum + rmsprop)
        avg_grad1 = 0; avg_grad2 = 0
        beta1 = 0.9; beta2 = 0.999; eps = 1e-07
        for itr in range(self.iterations):
            avg_grad1 = beta1 * avg_grad1 + (1 - beta1) * self.gradient_loss(pairs)
            avg_grad2 = (beta2 * avg_grad2 + (1 - beta2) * (self.gradient_loss(pairs) ** 2))
            avg_grad1_corr = avg_grad1 / (1 - beta1 ** (itr + 1))
            avg_grad2_corr = avg_grad2 / (1 - beta2 ** (itr + 1))
            self.w = self.w - self.lr * (avg_grad1_corr / (np.sqrt(avg_grad2_corr) + eps))

    # =====================================
    def update(self, u, yu):
        super().update(u, yu)
        pairs = generate_pairs(u, yu, self.scores, self.H, self.k, self.delta)
        self.minimize_w(pairs)
        self.scores = self.get_scores(self.Z)
