import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""


# -------------------------------------------------
# NORMALIZATION
# -------------------------------------------------

def normalize_image(image, mode="0to1"):

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy array")

    if image.size == 0:
        raise ValueError("image cannot be empty")

    img = image.astype(np.float32)

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return np.zeros_like(img)

    if mode in ["0to1", "minmax"]:

        return (img - min_val) / (max_val - min_val)

    elif mode in ["0to255", "uint8"]:

        normalized = (img - min_val) / (max_val - min_val)

        return normalized * 255

    elif mode == "zscore":

        mean = np.mean(img)
        std = np.std(img)

        if std == 0:
            return img - mean

        return (img - mean) / std

    else:

        raise ValueError(
            "mode must be: 0to1, 0to255, zscore"
        )


# -------------------------------------------------
# CLIPPING
# -------------------------------------------------

def clip_image(image, min_value=None, max_value=None):

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy array")

    if min_value is None or max_value is None:

        if image.dtype == np.uint8:

            min_value = 0
            max_value = 255

        else:

            min_value = 0.0
            max_value = 1.0

    return np.clip(image, min_value, max_value)


clip_pixels = clip_image


# -------------------------------------------------
# PADDING
# -------------------------------------------------

def pad_image(image, pad_size=1, mode="zero"):

    if isinstance(pad_size, int):

        pad_h = pad_size
        pad_w = pad_size

    else:

        pad_h, pad_w = pad_size


    if image.ndim == 2:

        pad_tuple = (
            (pad_h, pad_h),
            (pad_w, pad_w)
        )

    elif image.ndim == 3:

        pad_tuple = (
            (pad_h, pad_h),
            (pad_w, pad_w),
            (0,0)
        )

    else:

        raise ValueError("image must be 2D or 3D")


    if mode == "zero":

        return np.pad(
            image,
            pad_tuple,
            mode="constant",
            constant_values=0
        )

    elif mode == "reflect":

        return np.pad(image, pad_tuple, mode="reflect")

    elif mode == "edge":

        return np.pad(image, pad_tuple, mode="edge")

    else:

        raise ValueError("mode must be zero, reflect, or edge")


# -------------------------------------------------
# VECTORIZED 2D CONVOLUTION
# -------------------------------------------------

def convolve2d(image, kernel, padding_mode="zero"):

    if image.ndim != 2:
        raise ValueError("input must be grayscale image")

    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")

    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")


    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    kH, kW = kernel.shape

    pad_h = kH // 2
    pad_w = kW // 2


    padded = pad_image(
        image,
        (pad_h, pad_w),
        mode=padding_mode
    )


    # true convolution requires kernel flip
    kernel = np.flip(kernel)


    # create all sliding windows at once
    patches = sliding_window_view(
        padded,
        (kH, kW)
    )


    # vectorized multiply + sum
    output = np.einsum(
        'ijkl,kl->ij',
        patches,
        kernel
    )


    return output.astype(np.float32)


# -------------------------------------------------
# APPLY FILTER
# -------------------------------------------------

def apply_filter(image, kernel, padding_mode="zero"):

    if image.ndim == 2:

        return convolve2d(
            image,
            kernel,
            padding_mode
        )

    elif image.ndim == 3:

        result = np.zeros_like(
            image,
            dtype=np.float32
        )

        for c in range(image.shape[2]):

            result[:,:,c] = convolve2d(
                image[:,:,c],
                kernel,
                padding_mode
            )

        return result

    else:

        raise ValueError(
            "image must be 2D or 3D"
        )


# -------------------------------------------------
# TEST KERNEL
# -------------------------------------------------

mean_kernel = np.array(
[
    [1/9,1/9,1/9],
    [1/9,1/9,1/9],
    [1/9,1/9,1/9]
],
dtype=np.float32
)