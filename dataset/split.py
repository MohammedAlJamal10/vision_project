"""
dataset/split.py
================
Stratified train / validation / test split — implemented from scratch
using only NumPy and Pandas (no sklearn).

Stratification ensures that each class is represented in every split in
the same proportion as the full dataset.
"""

import numpy as np
from typing import Tuple


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Split arrays (X, y) into train / val / test sets stratified by y.

    Parameters
    ----------
    X : np.ndarray  (N, ...)
    y : np.ndarray  (N,)  integer class labels
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0 (±1e-9).
    seed : int

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n   = len(idx)
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_val   = max(1, int(np.floor(n * val_ratio)))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    # Shuffle each split
    train_idx = rng.permutation(train_idx)
    val_idx   = rng.permutation(val_idx)
    test_idx  = rng.permutation(test_idx)

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx],   y[val_idx]),
        (X[test_idx],  y[test_idx]),
    )


def split_summary(y_train, y_val, y_test, class_names=None):
    """Print a readable split summary."""
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = np.bincount(y)
        total  = len(y)
        print(f"\n{name} set — {total} samples")
        for i, c in enumerate(counts):
            label = class_names[i] if class_names else str(i)
            print(f"  {label:12s}: {c:4d}  ({100*c/total:.1f}%)")
