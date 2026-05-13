"""
dataset/augment.py
==================
Image augmentation pipeline — ALL transforms use minicv exclusively.

Implemented transforms (≥ 5 required)
--------------------------------------
1. random_rotation      — rotate by random angle in [−max_angle, +max_angle]
2. random_translation   — shift by random (tx, ty) in pixels
3. random_flip          — horizontal flip with probability p
4. brightness_scaling   — multiply intensities by random factor in [lo, hi]
5. gaussian_noise       — add zero-mean Gaussian noise
6. random_crop_resize   — random crop + resize back (adds scale/position invariance)
7. random_blur          — apply Gaussian blur with random sigma

All functions accept and return float64 images in [0,1].
"""


import numpy as np
import minicv as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def augment_cnn_batch(
    Xb: np.ndarray,
    rng: np.random.Generator,
    flip_p: float = 0.5,
    max_angle: float = 10.0,
) -> np.ndarray:
    """Safe CNN training augmentation for (N, C, 64, 64) batches."""
    X_aug = Xb.copy()

    for i in range(len(X_aug)):
        if X_aug.shape[1] == 1:
            img = X_aug[i, 0]

            if rng.random() < flip_p:
                img = cv.flip(img, axis="horizontal")

            angle = float(rng.uniform(-max_angle, max_angle))
            img = cv.rotate(img, angle, method="bilinear", fill_value=0.0)
            X_aug[i, 0] = np.clip(img, 0.0, 1.0)

        else:
            img = np.transpose(X_aug[i], (1, 2, 0))

            if rng.random() < flip_p:
                img = cv.flip(img, axis="horizontal")

            angle = float(rng.uniform(-max_angle, max_angle))
            img = cv.rotate(img, angle, method="bilinear", fill_value=0.0)
            X_aug[i] = np.transpose(np.clip(img, 0.0, 1.0), (2, 0, 1))

    return X_aug.astype(Xb.dtype, copy=False)


# ── Individual transforms ─────────────────────────────────────────────────────

def random_rotation(img: np.ndarray, max_angle: float = 30.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Rotate image by angle sampled uniformly from ±max_angle degrees."""
    rng   = rng or np.random.default_rng()
    angle = float(rng.uniform(-max_angle, max_angle))
    return cv.rotate(img, angle, method="bilinear", fill_value=0.0)


def random_translation(img: np.ndarray, max_shift: int = 8,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Translate image by a random integer (tx, ty) shift."""
    rng = rng or np.random.default_rng()
    tx  = int(rng.integers(-max_shift, max_shift + 1))
    ty  = int(rng.integers(-max_shift, max_shift + 1))
    return cv.translate(img, tx, ty, fill_value=0.0)


def random_flip(img: np.ndarray, p: float = 0.5,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip image horizontally with probability p."""
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        return cv.flip(img, axis="horizontal")
    return img.copy()


def brightness_scaling(img: np.ndarray, lo: float = 0.6, hi: float = 1.4,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Multiply all pixel values by a random factor in [lo, hi]."""
    rng    = rng or np.random.default_rng()
    factor = float(rng.uniform(lo, hi))
    return cv.clip_image(img * factor, 0.0, 1.0)


def gaussian_noise(img: np.ndarray, sigma: float = 0.05,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation sigma."""
    rng   = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float64)
    return cv.clip_image(img + noise, 0.0, 1.0)


def random_crop_resize(img: np.ndarray, crop_ratio: float = 0.85,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly crop a sub-region then resize back to original shape."""
    rng = rng or np.random.default_rng()
    H, W = img.shape[:2]
    ch   = int(H * crop_ratio); cw = int(W * crop_ratio)
    r0   = int(rng.integers(0, H - ch + 1))
    c0   = int(rng.integers(0, W - cw + 1))
    cropped = cv.crop(img, c0, r0, c0 + cw, r0 + ch)
    return cv.resize(cropped, H, W, method="bilinear")


def random_blur(img: np.ndarray, sigma_range: tuple = (0.3, 1.5),
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply Gaussian blur with sigma sampled uniformly from sigma_range."""
    rng   = rng or np.random.default_rng()
    sigma = float(rng.uniform(*sigma_range))
    size  = 5   # fixed kernel size; sigma controls strength
    return cv.gaussian_filter(img, size=size, sigma=sigma, padding_mode="reflect")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

ALL_TRANSFORMS = [
    random_rotation,
    random_translation,
    random_flip,
]
TRANSFORM_NAMES = [
    "Rotation", "Translation", "Flip",
]


def augment_image(
    img: np.ndarray,
    transforms: list | None = None,
    rng: np.random.Generator | None = None,
    p_each: float = 0.3,
) -> np.ndarray:
    """Apply a random subset of transforms to *img*.

    Each transform is applied with probability *p_each* (default 0.6).
    At least one transform is always applied.

    Parameters
    ----------
    img        : float64 image in [0,1]
    transforms : list of callables (default ALL_TRANSFORMS)
    rng        : random generator (created fresh if None)
    p_each     : probability of applying each individual transform

    Returns
    -------
    np.ndarray  augmented image, same shape as input
    """
    rng        = rng or np.random.default_rng()
    transforms = transforms or ALL_TRANSFORMS
    out        = img.astype(np.float64)
    applied    = 0

    for fn in transforms:
        if rng.random() < p_each:
            out     = fn(out, rng=rng)
            applied += 1

    if applied == 0:                          # guarantee at least one transform
        out = transforms[int(rng.integers(len(transforms)))](out, rng=rng)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data by *multiplier* times.

    Only call this on the TRAINING split.

    Parameters
    ----------
    X          : (N, H, W) or (N, H, W, C) float64 images
    y          : (N,) integer labels
    multiplier : how many augmented copies to add per original image
    seed       : reproducibility seed

    Returns
    -------
    X_aug : (N*(1+multiplier), ...) combined original + augmented
    y_aug : (N*(1+multiplier),)
    """
    rng      = np.random.default_rng(seed)
    X_aug_ls = [X]
    y_aug_ls = [y]

    for _ in range(multiplier):
        batch = np.stack([augment_image(X[i], rng=rng) for i in range(len(X))])
        X_aug_ls.append(batch)
        y_aug_ls.append(y)

    return np.concatenate(X_aug_ls, axis=0), np.concatenate(y_aug_ls, axis=0)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_augmentation_panel(
    img: np.ndarray,
    rng: np.random.Generator | None = None,
    save_path: str | None = None,
):
    """Show before/after panels for all 7 individual augmentation transforms."""
    rng = rng or np.random.default_rng(99)

    n_transforms = len(ALL_TRANSFORMS)
    fig, axes = plt.subplots(2, n_transforms + 1,
                             figsize=((n_transforms + 1) * 2.2, 4.5))

    def _show(ax, image, title, border_color=None):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color); spine.set_linewidth(2)

    # Row 0: original (repeated)
    for j in range(n_transforms + 1):
        _show(axes[0, j], img, "Original" if j == 0 else "")

    # Row 1: each transform
    _show(axes[1, 0], img, "Original")
    for j, (fn, name) in enumerate(zip(ALL_TRANSFORMS, TRANSFORM_NAMES)):
        aug = fn(img.copy(), rng=np.random.default_rng(rng.integers(9999)))
        _show(axes[1, j + 1], aug, name, border_color="steelblue")

    plt.suptitle("Augmentation: Before (top) vs After (bottom)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
        print(f"Augmentation panel saved → {save_path}")
    else:
        plt.show()
