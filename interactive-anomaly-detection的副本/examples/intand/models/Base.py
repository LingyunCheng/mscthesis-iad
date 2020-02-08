import numpy as np
import copy
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt


class Base:
    def __init__(self, Z, X=None, lr=0.1, iterations=2000, plot=False):
        self.Z = Z
        self.X = X
        self.lr = lr
        self.iterations = iterations
        self.plot = plot
        self.H = {-1: [], 1: []}
        self.scores = self.Z.sum(axis=1)####

        self.nb_anomalies = []
        self.queried_ids = []
        self.efforts = []

        self.fig = None

    # =====================================
    def get_masked_scores(self):
        labeled_ids, _ = self.get_labeled_ids()
        s_copy = copy.deepcopy(self.scores)
        s_copy[labeled_ids] = -np.inf
        return s_copy

    # =====================================
    def get_labeled_ids(self):
        ids_labeled = self.H[-1] + self.H[1]
        labels = [-1 for _ in self.H[-1]] + [1 for _ in self.H[1]]
        return ids_labeled, labels

    # =====================================
    def get_top1(self):
        scores = self.get_masked_scores()
        i = np.argmax(scores)
        return i

    # =====================================
    def get_precision(self, n_anomalies):
        return [nb * 100 / n_anomalies for nb in self.nb_anomalies]

    # =====================================
    def update_expert_effort(self, u):
        self.queried_ids.append(u)

        if len(self.queried_ids) < 2:
            self.efforts.append(0)
        else:
            i, j = self.queried_ids[-1], self.queried_ids[-2]
            e = cosine_distance(self.X[i], self.X[j])
            self.efforts.append(self.efforts[-1] + e)

    # =====================================
    def get_expert_effort(self):
        return self.efforts

    # =====================================
    def update_plot(self):
        if self.fig is None:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
            self.ax2.set_ylabel("Nbr. of Anomalies Detected")
            self.ax2.set_xlabel("Nbr. of Feedbacks (Budget)")
            self.ax3.set_ylabel("Expert Effort")
            self.ax3.set_xlabel("Nbr. of Feedbacks (Budget)")

        self.ax1.scatter(self.X[:, 0], self.X[:, 1], marker=".", c=self.scores)
        self.ax1.scatter(self.X[self.H[-1], 0], self.X[self.H[-1], 1], marker="$-$", color="red")
        self.ax1.scatter(self.X[self.H[1], 0], self.X[self.H[1], 1], marker="$+$", color="red")
        self.ax2.plot(self.nb_anomalies, marker=".", color="red")
        self.ax3.plot(self.efforts, marker=".", color="red")

        self.fig.canvas.draw()
        plt.pause(0.05)

    # =====================================
    def update(self, u, yu):
        self.H[yu].append(u)
        self.nb_anomalies.append(len(self.H[1]))
        self.update_expert_effort(u)

        print("\rTotal: {} ({} anomalies, {} nominals)".format(len(self.H[1] + self.H[-1]), len(self.H[1]), len(self.H[-1])), end="")
        if self.plot:
            self.update_plot()
