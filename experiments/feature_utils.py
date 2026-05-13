import numpy as np

from features_pipeline.extractor import extract_features_batch


def grayscale_images(X):

    if X.ndim == 3:
        return X.astype(np.float64)

    return (
        0.2989 * X[..., 0]
        + 0.5870 * X[..., 1]
        + 0.1140 * X[..., 2]
    ).astype(np.float64)


def pooled_pixel_features(X, output_size=24):

    gray = grayscale_images(X)
    height, width = gray.shape[1:3]

    if height % output_size != 0 or width % output_size != 0:
        raise ValueError("Image size must be divisible by output_size")

    block_h = height // output_size
    block_w = width // output_size

    pooled = gray.reshape(
        len(gray),
        output_size,
        block_h,
        output_size,
        block_w
    ).mean(axis=(2, 4))

    return pooled.reshape(len(gray), -1)


def enhanced_fer_features(X, verbose=False):

    pixel_features = pooled_pixel_features(X, output_size=24)
    handcrafted_features = extract_features_batch(X, verbose=verbose)

    return np.concatenate(
        [pixel_features, handcrafted_features],
        axis=1
    )


def standardize_from_train(X_train, X_val, X_test):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
    )


def fit_pca(X_train, n_components):

    mean = X_train.mean(axis=0)
    centered = X_train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    return mean, vt[:n_components].T


def transform_pca(X, mean, components):

    return (X - mean) @ components


def pca_from_train(X_train, X_val, X_test, n_components):

    mean, components = fit_pca(X_train, n_components)

    return (
        transform_pca(X_train, mean, components),
        transform_pca(X_val, mean, components),
        transform_pca(X_test, mean, components),
    )


def add_quadratic_features(*splits):

    return tuple(
        np.concatenate([split, split**2], axis=1)
        for split in splits
    )
