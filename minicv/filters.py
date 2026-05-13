import numpy as np
import utils as fdf



"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""


# -----------------------------------
# Mean / Box Filter
# -----------------------------------
def mean_filter(image, kernel_size=3):
    """
    Apply a mean (box) filter to an image.

    Parameters:
        image (numpy.ndarray): Input grayscale or RGB image.
        kernel_size (int): Size of square averaging kernel.

    Returns:
        numpy.ndarray: Smoothed image.

    Raises:
        TypeError: If image is not a NumPy array or kernel_size is not an integer.
        ValueError: If kernel_size is not positive or not odd.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")

    if kernel_size <= 0:
        raise ValueError("kernel_size must be greater than zero.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return fdf.apply_filter(image, kernel)


def gaussian_kernel(size=3, sigma=1.0):
    """
    Generate a Gaussian kernel.

    Parameters:
        size (int): Kernel size. Should be odd.
        sigma (float): Standard deviation of Gaussian.

    Returns:
        numpy.ndarray: Normalized Gaussian kernel.

    Raises:
        TypeError: If size is not an integer or sigma is not numeric.
        ValueError: If size is invalid or sigma is not positive.
    """

    if not isinstance(size, int):
        raise TypeError("size must be an integer.")

    if not isinstance(sigma, (int, float)):
        raise TypeError("sigma must be numeric.")

    if size <= 0:
        raise ValueError("size must be greater than zero.")

    if size % 2 == 0:
        raise ValueError("size must be odd.")

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel


def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian filtering to an image.

    Parameters:
        image (numpy.ndarray): Input grayscale or RGB image.
        kernel_size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation of Gaussian.

    Returns:
        numpy.ndarray: Smoothed image.

    Raises:
        TypeError: If image is not a NumPy array.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    kernel = gaussian_kernel(kernel_size, sigma)
    return fdf.apply_filter(image, kernel)


# -----------------------------------
# Median Filter
# -----------------------------------
def median_filter(image, kernel_size=3):
    """
    Apply a median filter to a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        kernel_size (int): Size of square neighborhood.

    Returns:
        numpy.ndarray: Filtered image with reduced salt-and-pepper noise.

    Raises:
        TypeError: If image is not a NumPy array or kernel_size is not an integer.
        ValueError: If image is not grayscale or kernel_size is invalid.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("median_filter only supports grayscale images.")

    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")

    if kernel_size <= 0:
        raise ValueError("kernel_size must be greater than zero.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    pad_size = kernel_size // 2
    padded = np.pad(image, pad_size, mode="edge")
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(region)

    return output


# -----------------------------------
# Thresholding
# -----------------------------------
# -----------------------------------
# Thresholding
# -----------------------------------
def global_threshold(image, threshold=0.5):
    """
    Apply global thresholding to an image.

    Works with normalized images [0,1] or intensity images [0,255].

    Mathematical definition:

        g(x,y) = 1  if I(x,y) >= T
                 0  otherwise

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        threshold (int or float):
            Threshold value.
            If image in [0,1] → use threshold in [0,1]
            If image in [0,255] → threshold automatically scaled

    Returns
    -------
    np.ndarray

        binary image
        dtype : float32
        range : {0,1}

    Raises:
        TypeError:
            If image is not numpy array.
        ValueError:
            If image is not grayscale.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("global_threshold only supports grayscale images.")

    # automatically adapt threshold range
    if image.max() > 1.0:
        threshold = threshold / 255.0

    binary = np.where(image >= threshold, 1.0, 0.0)

    return binary


def otsu_threshold(image):
    """
    Apply Otsu thresholding to automatically determine
    the optimal threshold value.

    Otsu method maximizes between-class variance:

        σ_b² = w1 * w2 * (μ1 - μ2)^2

    where:
        w1,w2 → class probabilities
        μ1,μ2 → class means

    Steps:
        1. convert image to [0,255]
        2. compute histogram
        3. compute between-class variance
        4. select best threshold
        5. apply global threshold

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray:
            Binary image

    Raises:
        TypeError:
            If image is not numpy array.
        ValueError:
            If image is not grayscale.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("otsu_threshold only supports grayscale images.")

    # convert image to 8-bit range
    img = fdf.normalize_image(image, "0to255").astype(np.uint8)

    hist = np.zeros(256)

    for val in img.flatten():
        hist[val] += 1


    total_pixels = img.size

    sum_total = np.sum(
        np.arange(256) * hist
    )

    sum_background = 0
    weight_background = 0

    best_variance = 0
    best_threshold = 0


    for t in range(256):

        weight_background += hist[t]

        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background

        if weight_foreground == 0:
            break

        sum_background += t * hist[t]

        mean_background = sum_background / weight_background

        mean_foreground = (
            sum_total - sum_background
        ) / weight_foreground


        between_class_variance = (
            weight_background *
            weight_foreground *
            (mean_background - mean_foreground)**2
        )


        if between_class_variance > best_variance:

            best_variance = between_class_variance

            best_threshold = t


    return global_threshold(
        image,
        best_threshold / 255.0
    )


def adaptive_threshold(image, block_size=3, c=5):
    """
    Apply adaptive thresholding using local mean values.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        block_size (int): Size of local neighborhood.
        c (int or float): Constant subtracted from local mean.

    Returns
    -------
    np.ndarray
    binary image
    values : {0,255}

    Raises:
        TypeError: If input types are invalid.
        ValueError: If image is not grayscale or block_size is invalid.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("adaptive_threshold only supports grayscale images.")

    if not isinstance(block_size, int):
        raise TypeError("block_size must be an integer.")

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")

    if not isinstance(c, (int, float)):
        raise TypeError("c must be numeric.")

    pad_size = block_size // 2
    padded = np.pad(image, pad_size, mode="reflect")
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + block_size, j:j + block_size]
            local_mean = np.mean(region)

            if image[i, j] > local_mean - c:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


# -----------------------------------
# Sobel Gradients
# -----------------------------------
def sobel_gradients(image):
    """
    Compute Sobel gradients in x and y directions.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns
    -------
    gx : np.ndarray
     horizontal gradient

    gy : np.ndarray
     vertical gradient

    magnitude : np.ndarray

     sqrt(gx^2 + gy^2)

    dtype : float32

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If image is not grayscale.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("sobel_gradients only supports grayscale images.")

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    gx = fdf.convolve2d(image, sobel_x)
    gy = fdf.convolve2d(image, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)

    return gx, gy, magnitude


# -----------------------------------
# Bit Plane Slicing
# -----------------------------------
def bit_plane_slice(image, bit):
    """
    Extract a specific bit plane from an image.

    Image is first converted to 8-bit representation.

    Bit-plane slicing decomposes image:

        pixel = b7*2^7 + b6*2^6 + ... + b0*2^0

    each bit plane highlights different image structures.

    MSB (bit 7):
        captures major shapes

    LSB (bit 0):
        captures fine noise details

    Parameters:
        image (numpy.ndarray):
            Input grayscale image

        bit (int):
            bit index 0 → 7

    Returns:
        numpy.ndarray:
            binary image (0 or 1)

    Raises:
        TypeError:
            invalid input types
        ValueError:
            invalid bit index or image shape
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("bit_plane_slice only supports grayscale images.")

    if not isinstance(bit, int):
        raise TypeError("bit must be integer.")

    if bit < 0 or bit > 7:
        raise ValueError("bit must be between 0 and 7.")

    img = fdf.normalize_image(
        image,
        "0to255"
    ).astype(np.uint8)

    plane = ((img >> bit) & 1).astype(np.float32)

    return plane

# -----------------------------------
# Histogram
# -----------------------------------
# -----------------------------------
# Histogram
# -----------------------------------
def histogram(image):
    """
    Compute histogram of grayscale intensities.

    Histogram counts number of pixels per intensity:

        h(r_k) = number of pixels with intensity r_k

    Image automatically converted to [0,255].

    Parameters:
        image (numpy.ndarray):
            grayscale image

    Returns
    -------
    np.ndarray

        contrast enhanced image

        dtype : float32

        range : [0,1]
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("histogram only supports grayscale images.")

    img = fdf.normalize_image(
        image,
        "0to255"
    ).astype(np.uint8)

    hist = np.zeros(256)

    for value in img.flatten():

        hist[value] += 1

    return hist

# -----------------------------------
# Histogram Equalization
# -----------------------------------
def histogram_equalization(image):
    """
    Improve image contrast using histogram equalization.

    Transformation:

        s = (L-1) * CDF(r)

    where:
        CDF = cumulative distribution function

    Steps:
        1. compute histogram
        2. compute cumulative distribution
        3. remap intensities

    Parameters:
        image (numpy.ndarray):
            grayscale image

    Returns:
        numpy.ndarray:
            contrast enhanced image in range [0,1]
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("histogram_equalization only supports grayscale images.")

    img = fdf.normalize_image(
        image,
        "0to255"
    ).astype(np.uint8)

    hist = histogram(img)

    cdf = np.cumsum(hist)

    cdf = cdf / cdf[-1]

    equalized = cdf[img]

    return equalized.astype(np.float32)

# -----------------------------------
# Additional Technique 1: Laplacian
# -----------------------------------
def laplacian_filter(image):
    """
    Apply Laplacian filter for edge detection.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Edge-enhanced image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("laplacian_filter only supports grayscale images.")

    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    return fdf.convolve2d(image, kernel)


# -----------------------------------
# Additional Technique 2: Sharpening
# -----------------------------------
def sharpening_filter(image):
    """
    Apply sharpening filter to an image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Sharpened image.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("sharpening_filter only supports grayscale images.")

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return fdf.convolve2d(image, kernel)