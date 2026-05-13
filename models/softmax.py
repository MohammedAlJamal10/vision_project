"""
models/softmax.py
=================
Multinomial Softmax Regression trained with mini-batch gradient descent.

Forward pass
------------
    z = X·W + b
    P(y=i|x) = exp(zᵢ) / Σⱼ exp(zⱼ)         (numerically stable)

Loss (cross-entropy with ε clipping)
--------------------------------------
    L = −(1/N) Σ log( p[y] + ε )

Gradient
--------
    ∂L/∂W = (1/N) Xᵀ · (P − Y_onehot)
    ∂L/∂b = (1/N) Σ (P − Y_onehot)

L2 regularisation
-----------------
    L_reg = L + (λ/2) ||W||²
    ∂L_reg/∂W = ∂L/∂W + λ·W
"""

import numpy as np


class SoftmaxRegression:
    """Multinomial softmax regression from scratch.

    Parameters
    ----------
    n_features  : int
    n_classes   : int
    lr          : float   initial learning rate
    lambda_l2   : float   L2 weight decay
    batch_size  : int
    max_epochs  : int
    eps         : float   cross-entropy clipping constant
    """

    def __init__(
        self,
        n_features:  int,
        n_classes:   int,
        lr:          float = 1e-2,
        lambda_l2:   float = 1e-4,
        batch_size:  int   = 64,
        max_epochs:  int   = 200,
        eps:         float = 1e-9,
    ):
        self.n_features  = n_features
        self.n_classes   = n_classes
        self.lr          = lr
        self.lambda_l2   = lambda_l2
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.eps         = eps

        # Xavier initialisation
        scale = np.sqrt(2.0 / (n_features + n_classes))
        self.W = np.random.default_rng(0).normal(0, scale, (n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=np.float64)

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax (subtract row-max before exp)."""
        z  = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        return ex / ex.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)

    # ── Loss & gradient ───────────────────────────────────────────────────────

    def _cross_entropy(self, P: np.ndarray, y: np.ndarray) -> float:
        N    = len(y)
        p_y  = P[np.arange(N), y]
        loss = -np.mean(np.log(p_y + self.eps))
        loss += (self.lambda_l2 / 2) * np.sum(self.W ** 2)
        return float(loss)

    def _gradients(self, X: np.ndarray, P: np.ndarray, y: np.ndarray):
        N   = len(y)
        dZ  = P.copy()
        dZ[np.arange(N), y] -= 1.0         # (P − Y_onehot)
        dW  = (X.T @ dZ) / N + self.lambda_l2 * self.W
        db  = dZ.mean(axis=0)
        return dW, db

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        optimizer       = None,
        scheduler       = None,
        early_stopping  = None,
        logger          = None,
        grad_clip_norm: float | None = 5.0,
        early_stopping_metric: str = "val_loss",
    ) -> "SoftmaxRegression":
        """Train with mini-batch gradient descent.

        Parameters
        ----------
        optimizer      : SGD or Adam instance (uses self.lr if None)
        scheduler      : LRScheduler instance
        early_stopping : EarlyStopping instance
        logger         : TrainingLogger instance
        grad_clip_norm : max gradient L2 norm (None = no clipping)
        """
        N   = len(y_train)
        rng = np.random.default_rng(42)

        for epoch in range(1, self.max_epochs + 1):
            # ── shuffle ──────────────────────────────────────────────────────
            perm = rng.permutation(N)
            X_sh, y_sh = X_train[perm], y_train[perm]

            # ── mini-batch updates ────────────────────────────────────────────
            for start in range(0, N, self.batch_size):
                Xb = X_sh[start:start + self.batch_size]
                yb = y_sh[start:start + self.batch_size]
                P  = self._forward(Xb)
                dW, db = self._gradients(Xb, P, yb)

                # gradient clipping
                if grad_clip_norm is not None:
                    gnorm = np.sqrt(np.sum(dW**2) + np.sum(db**2))
                    if gnorm > grad_clip_norm:
                        scale = grad_clip_norm / gnorm
                        dW *= scale; db *= scale

                if optimizer is not None:
                    self.W, self.b = optimizer.step(
                        self.W, self.b, dW, db, key="softmax"
                    )
                else:
                    self.W -= self.lr * dW
                    self.b -= self.lr * db

            # ── epoch metrics ─────────────────────────────────────────────────
            P_tr = self._forward(X_train)
            P_va = self._forward(X_val)
            tr_loss = self._cross_entropy(P_tr, y_train)
            va_loss = self._cross_entropy(P_va, y_val)
            tr_acc  = float((P_tr.argmax(1) == y_train).mean())
            va_acc  = float((P_va.argmax(1) == y_val).mean())
            current_lr = optimizer.lr if optimizer else self.lr

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)
            self.history["lr"].append(current_lr)

            if logger:
                logger.log(epoch, tr_loss, va_loss, tr_acc, va_acc, current_lr)
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d} | tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f}"
                      f" | tr_acc={tr_acc:.4f} | va_acc={va_acc:.4f}")

            # ── learning rate schedule ────────────────────────────────────────
            if scheduler:
                new_lr = scheduler.step(epoch, va_loss)
                if optimizer:
                    optimizer.lr = new_lr

            # ── early stopping ────────────────────────────────────────────────
            if early_stopping:
                params = {"W": self.W, "b": self.b}
                if early_stopping_metric == "val_acc":
                    stop_value = -va_acc
                else:
                    stop_value = va_loss

                if early_stopping.step(stop_value, params):
                    print(f"  Early stopping at epoch {epoch}")
                    best = early_stopping.best_params
                    self.W, self.b = best["W"], best["b"]
                    break

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X.astype(np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ── Serialisation ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict):
        self.W = params["W"]; self.b = params["b"]
