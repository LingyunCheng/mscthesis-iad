from . import Base


class Unsupervised(Base):
    def __init__(self, Z, X=None, lr=0.1, iterations=2000, plot=False):
        super().__init__(Z, X, lr, iterations, plot)
        self.scores = self.get_scores(self.Z)

    # =====================================
    def get_scores(self, Z):
        return Z.sum(axis=1)

    # =====================================
    def update(self, u, yu):
        super().update(u, yu)
