import numpy as np
import matplotlib.pyplot as plt
from typing import Union


"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""



def read_image(path: str, as_gray: bool = False, normalize: bool = True) -> np.ndarray:
    """
    Read an image from disk into a NumPy array.

    Parameters
    ----------
    path : str
        Path to image file.
    as_gray : bool, optional
        If True, convert image to grayscale.
    normalize : bool, optional
        If True, scale pixel values to range [0,1].

    Returns
    -------
    np.ndarray
        Image array with shape:
            grayscale → (H,W)
            RGB → (H,W,3)

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If image format unsupported.
        
        
    Notes
    -----
    Output follows MiniCV convention:

    dtype  : float32
    range  : [0,1]
    """

    try:
        img = plt.imread(path)
    except Exception as e:
        raise FileNotFoundError(f"Unable to read image at {path}") from e

    img = _ensure_float(img)

    if as_gray:
        img = rgb_to_gray(img)

    if normalize:
        img = normalize_image(img, mode="minmax")

    return img


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save NumPy image array to disk.

    Parameters
    ----------
    path : str
        Output file path (.png or .jpg)
    image : np.ndarray
        Image array (grayscale or RGB)

    Raises
    ------
    ValueError
        If array shape invalid.
        
    Notes
    -----
    Input expected in range [0,1].
    Values automatically clipped.    
    """

    _validate_image_array(image)

    img = np.clip(image, 0, 1)

    plt.imsave(path, img, cmap="gray" if image.ndim == 2 else None)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using luminosity method.

    Y = 0.299R + 0.587G + 0.114B

    Parameters
    ----------
    image : np.ndarray
        RGB image (H,W,3)

    Returns
    -------
    np.ndarray
        Grayscale image (H,W)
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H,W,3)")

    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    gray = 0.299*r + 0.587*g + 0.114*b

    return gray


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H,W)

    Returns
    -------
    np.ndarray
        RGB image (H,W,3)
    """

    if image.ndim != 2:
        raise ValueError("Input must be grayscale image")

    return np.stack([image, image, image], axis=2)


def normalize_image(image: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    Normalize image values.

    Parameters
    ----------
    image : np.ndarray

    mode : str
        'minmax' → scale to [0,1]
        'zscore' → zero mean unit variance
        'uint8' → scale to [0,255]

    Returns
    -------
    np.ndarray
    """

    img = image.astype(np.float32)

    if mode == "minmax":

        min_val = np.min(img)
        max_val = np.max(img)

        if max_val - min_val == 0:
            return img

        return (img - min_val) / (max_val - min_val)

    elif mode == "zscore":

        mean = np.mean(img)
        std = np.std(img)

        if std == 0:
            return img - mean

        return (img - mean) / std

    elif mode == "uint8":

        img = normalize_image(img, "minmax")

        return (img * 255).astype(np.uint8)

    else:

        raise ValueError("mode must be minmax, zscore, or uint8")


def clip_image(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Clip pixel intensities.

    Parameters
    ----------
    image : np.ndarray
    min_val : float
    max_val : float

    Returns
    -------
    np.ndarray
    """

    return np.clip(image, min_val, max_val)


def _validate_image_array(image: np.ndarray) -> None:

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy array")

    if image.ndim not in (2,3):
        raise ValueError("image must be 2D or 3D array")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("RGB image must have 3 channels")


def _ensure_float(image: np.ndarray) -> np.ndarray:

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255

    return image.astype(np.float32)