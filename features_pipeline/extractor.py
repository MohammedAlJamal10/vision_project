"""
features/extractor.py
=====================
Extracts a rich feature vector from each image using ONLY minicv.

Feature Families (≥ 3 required)
---------------------------------
Family A — Intensity Histogram (64 bins)            dim = 64
Family B — Gradient Magnitude Histogram (32 bins)   dim = 32
Family C — Bit-Plane Statistics (8 planes × 2)      dim = 16
Family D — Global Statistics                        dim =  6
           [mean, std, skewness, kurtosis,
            otsu_threshold, edge_density]
Family E — LBP-like Texture (uniform 2-bin per cell) dim = 16×16×2 = 512
           → encoded as 16 histogram bins (compact form)   dim = 16

Total dimensionality: 64 + 32 + 16 + 6 + 16 = 134

Feature index ranges (used by MRMR and models):
  [0   :64 ] — intensity histogram
  [64  :96 ] — gradient magnitude histogram
  [96  :112] — bit-plane mean/std pairs
  [112 :118] — global statistics
  [118 :134] — compact texture histogram
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import minicv as cv

# ── Feature dimension constants ───────────────────────────────────────────────
HIST_BINS   = 64
GRAD_BINS   = 32
N_BITPLANES = 8
N_GLOBAL    = 6
TEX_BINS    = 16
FEATURE_DIM = HIST_BINS + GRAD_BINS + N_BITPLANES * 2 + N_GLOBAL + TEX_BINS  # 134

FEATURE_NAMES = (
    [f"hist_{i}" for i in range(HIST_BINS)]
    + [f"grad_{i}" for i in range(GRAD_BINS)]
    + [f"bp{b}_{s}" for b in range(N_BITPLANES) for s in ("mean", "std")]
    + ["mean", "std", "skewness", "kurtosis", "otsu_t", "edge_density"]
    + [f"tex_{i}" for i in range(TEX_BINS)]
)
assert len(FEATURE_NAMES) == FEATURE_DIM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_gray(img: np.ndarray) -> np.ndarray:
    """Return a float64 grayscale image in [0,1]."""
    gray = cv.rgb_to_gray(img) if img.ndim == 3 else img.astype(np.float64)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def _moment(x: np.ndarray, order: int, mean: float, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mean) / std) ** order))


def _compact_texture(img: np.ndarray) -> np.ndarray:
    """
    Simplified local texture descriptor using gradient direction histogram.

    The gradient direction (in radians, range [−π, π]) is quantised into
    TEX_BINS uniform bins and normalised → a compact orientation histogram
    that captures dominant edge directions.
    """
    gx, gy = cv.sobel_gradients(img)
    angles  = np.arctan2(gy, gx)                        # (H, W) in [-π, π]
    h, _    = np.histogram(angles.ravel(), bins=TEX_BINS, range=(-np.pi, np.pi))
    total   = h.sum()
    return (h / total if total > 0 else h).astype(np.float64)


# ── Core extractor ────────────────────────────────────────────────────────────

def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract the full 134-D feature vector from a single image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) float64 image in [0, 1].

    Returns
    -------
    np.ndarray
        1-D float64 vector of length FEATURE_DIM (134).
    """
    gray = _safe_gray(img)

    # ── A: Intensity histogram (64) ──────────────────────────────────────────
    hist_feat = cv.histogram_descriptor(gray, bins=HIST_BINS)

    # ── B: Gradient magnitude histogram (32) ────────────────────────────────
    grad_feat = cv.gradient_magnitude_histogram(gray, bins=GRAD_BINS)

    # ── C: Bit-plane mean + std (16) ─────────────────────────────────────────
    planes    = cv.all_bit_planes(gray)           # list of 8 binary (H,W) uint8
    bp_stats  = np.array(
        [[p.mean(), p.std()] for p in planes], dtype=np.float64
    ).ravel()                                      # (16,)

    # ── D: Global statistics (6) ──────────────────────────────────────────────
    flat   = gray.ravel().astype(np.float64)
    mu     = float(flat.mean())
    sigma  = float(flat.std())
    skew   = _moment(flat, 3, mu, sigma)
    kurt   = _moment(flat, 4, mu, sigma)
    _, ot  = cv.otsu_threshold(gray)
    ed     = cv.edge_density(gray, threshold=0.05)
    global_feat = np.array([mu, sigma, skew, kurt, ot, ed], dtype=np.float64)

    # ── E: Compact texture / orientation histogram (16) ──────────────────────
    tex_feat = _compact_texture(gray)

    feat = np.concatenate([hist_feat, grad_feat, bp_stats, global_feat, tex_feat])
    assert len(feat) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feat)}"
    return feat


def extract_features_batch(
    X: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Extract features from a batch of images.

    Parameters
    ----------
    X : np.ndarray  (N, H, W) or (N, H, W, 3)
    verbose : bool

    Returns
    -------
    np.ndarray  (N, FEATURE_DIM)
    """
    N    = len(X)
    feats = np.zeros((N, FEATURE_DIM), dtype=np.float64)
    for i in range(N):
        feats[i] = extract_features(X[i])
        if verbose and (i + 1) % 100 == 0:
            print(f"  Extracted {i+1}/{N} feature vectors …")
    return feats