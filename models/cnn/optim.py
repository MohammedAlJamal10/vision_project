"""
models/cnn/optim.py
===================

SGD + Adam optimizers for NumPy CNNs.
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent with optional momentum.
    """

    def __init__(self, lr=0.01, momentum=0.0):

        self.lr = lr
        self.momentum = momentum

        self.vW = {}
        self.vb = {}

    def step(self, W, b, dW, db, key=None, update_t=True):

        key = key if key is not None else id(W)

        # ── Momentum SGD ─────────────────────
        if self.momentum > 0:

            if key not in self.vW:
                self.vW[key] = np.zeros_like(W)
                self.vb[key] = np.zeros_like(b)

            self.vW[key] = (
                self.momentum * self.vW[key]
                - self.lr * dW
            )

            self.vb[key] = (
                self.momentum * self.vb[key]
                - self.lr * db
            )

            W_new = W + self.vW[key]
            b_new = b + self.vb[key]

        # ── Plain SGD ────────────────────────
        else:

            W_new = W - self.lr * dW
            b_new = b - self.lr * db

        return W_new, b_new


class Adam:
    """
    Adam optimizer.

    Uses per-parameter moment estimates.
    """

    def __init__(
        self,
        lr=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        # ── First moments ────────────────────
        self.mW = {}
        self.mb = {}

        # ── Second moments ───────────────────
        self.vW = {}
        self.vb = {}

    def step(self, W, b, dW, db, key=None, update_t=True):

        if update_t:
            self.t += 1

        keyW = ("W", key if key is not None else id(W))
        keyB = ("b", key if key is not None else id(b))

        # ── Initialize states ────────────────

        if keyW not in self.mW:

            self.mW[keyW] = np.zeros_like(W)
            self.vW[keyW] = np.zeros_like(W)

        if keyB not in self.mb:

            self.mb[keyB] = np.zeros_like(b)
            self.vb[keyB] = np.zeros_like(b)

        # ── Update first moments ─────────────

        self.mW[keyW] = (
            self.beta1 * self.mW[keyW]
            + (1 - self.beta1) * dW
        )

        self.mb[keyB] = (
            self.beta1 * self.mb[keyB]
            + (1 - self.beta1) * db
        )

        # ── Update second moments ────────────

        self.vW[keyW] = (
            self.beta2 * self.vW[keyW]
            + (1 - self.beta2) * (dW ** 2)
        )

        self.vb[keyB] = (
            self.beta2 * self.vb[keyB]
            + (1 - self.beta2) * (db ** 2)
        )

        # ── Bias correction ──────────────────

        mW_hat = self.mW[keyW] / (1 - self.beta1 ** self.t)
        mb_hat = self.mb[keyB] / (1 - self.beta1 ** self.t)

        vW_hat = self.vW[keyW] / (1 - self.beta2 ** self.t)
        vb_hat = self.vb[keyB] / (1 - self.beta2 ** self.t)

        # ── Parameter update ─────────────────

        W_new = (
            W
            - self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        )

        b_new = (
            b
            - self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        )

        return W_new, b_new
