"""
models/knn.py
=============

KNN with:
- correct distance computation
- optional distance weighting (stable)
- k-sweep
"""

import numpy as np


class KNNClassifier:

    def __init__(self, k: int = 5, weighted: bool = False):
        if k < 1:
            raise ValueError("k must be >= 1")

        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        self.n_classes = None

    # ─────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X.astype(np.float64)
        self.y_train = y.astype(np.int64)
        self.n_classes = int(y.max()) + 1
        return self

    # ─────────────────────────────────────────────
    # Distance
    # ─────────────────────────────────────────────
    def _euclidean_distances(self, X: np.ndarray):

        X = X.astype(np.float64)

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
        X2 = np.sum(X**2, axis=1, keepdims=True)
        Y2 = np.sum(self.X_train**2, axis=1)
        XY = X @ self.X_train.T

        dist2 = X2 + Y2 - 2 * XY
        dist2 = np.clip(dist2, 0, None)

        return np.sqrt(dist2)

    # ─────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────
    def predict(self, X: np.ndarray):

        if self.X_train is None:
            raise RuntimeError("Model not fitted")

        dists = self._euclidean_distances(X)

        # Get k nearest
        k = min(self.k, len(self.X_train))
        nn_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        nn_labels = self.y_train[nn_idx]

        # --- UNWEIGHTED ---
        if not self.weighted:
            preds = np.array([
                np.bincount(row, minlength=self.n_classes).argmax()
                for row in nn_labels
            ])
            return preds

        # --- WEIGHTED ---
        nn_dists = dists[np.arange(len(dists))[:, None], nn_idx]

        # Avoid division explosion
        weights = 1.0 / (nn_dists + 1e-6)

        preds = []

        for i in range(len(nn_labels)):
            votes = np.zeros(self.n_classes)

            for j in range(k):
                label = nn_labels[i, j]
                votes[label] += weights[i, j]

            preds.append(np.argmax(votes))

        return np.array(preds)

    # ─────────────────────────────────────────────
    # Score
    # ─────────────────────────────────────────────
    def score(self, X: np.ndarray, y: np.ndarray):
        return float((self.predict(X) == y).mean())


# ─────────────────────────────────────────────
# k sweep
# ─────────────────────────────────────────────
def sweep_k(
    X_train,
    y_train,
    X_val,
    y_val,
    k_values=None,
    weighted=False
):

    k_values = k_values or [3,5,7,9,11,13,15,17,19,21]
    results = {}

    for k in k_values:
        model = KNNClassifier(k=k, weighted=weighted).fit(X_train, y_train)
        acc = model.score(X_val, y_val)
        results[k] = acc
        print(f"k={k:3d}  val_acc={acc:.4f}")

    best_k = max(results, key=results.get)
    print(f"\nBest k = {best_k} (val_acc = {results[best_k]:.4f})")

    return best_k, results
