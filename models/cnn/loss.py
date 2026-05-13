"""
models/cnn/loss.py  — FIXED
============================

Bug fixed: double-division by N
---------------------------------
The original backward() returned ``dz / N``.  The Linear and Conv2D
backward passes ALSO divide by N (``dW = X.T @ dout / N``).

Because these divisions compound, the actual gradient delivered to each
weight was:

    dW_actual = dW_correct / N²   (with N = batch_size = 32 → 1024× too small)

This makes the effective learning rate 1024× smaller than the user-specified
value, causing extremely slow convergence and the erratic behaviour seen
in the training curves.

Fix: backward() returns ``dz`` (NOT ``dz / N``).  The /N normalisation is
applied once inside each layer's backward (as it already was).  The loss
value itself is still a proper mean — only the *gradient* convention changes.
"""

import numpy as np


class SoftmaxCrossEntropyLoss:
    """Numerically-stable softmax cross-entropy loss.

    Convention (after fix)
    ----------------------
    forward()  → scalar mean loss (divided by N for reporting)
    backward() → dL/d(logits) of shape (N, C), NOT divided by N
                 Each layer's backward then divides by N when computing dW.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps   = eps
        self._prob = None
        self._y    = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        z    = logits - logits.max(axis=1, keepdims=True)   # numerical stability
        ex   = np.exp(z)
        prob = ex / ex.sum(axis=1, keepdims=True)
        self._prob = prob
        self._y    = y
        N      = len(y)
        p_y    = prob[np.arange(N), y]
        loss   = -np.mean(np.log(p_y + self.eps))           # mean over batch
        return float(loss)

    def backward(self) -> np.ndarray:
        """Return dL/d(logits), shape (N, C).

        NOTE: does NOT divide by N here.
        The consuming layer's backward (Linear, Conv2D) applies the /N.
        This gives a single clean /N in the entire graph.
        """
        N  = len(self._y)
        dz = self._prob.copy()
        dz[np.arange(N), self._y] -= 1.0
        return dz                                            # ← removed / N

    @property
    def probabilities(self) -> np.ndarray:
        return self._prob