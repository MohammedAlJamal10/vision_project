import numpy as np
import matplotlib.pyplot as mp
import matplotlib.image as mimage
import utils as fdf



"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""



def resize_image(image, scaleX, scaleY, method="bilinear"):
    """
    Resize an image using geometric scaling and interpolation.

    Supported interpolation methods:

        nearest   → nearest neighbor interpolation (fast, blocky)
        bilinear  → bilinear interpolation (smooth, default)

    Mapping between coordinate systems:

        x_old = x_new * (W_old / W_new)
        y_old = y_new * (H_old / H_new)

    where:
        (x_new, y_new) are pixel coordinates in resized image
        (x_old, y_old) are corresponding coordinates in original image

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale 2D or RGB 3D)

    scaleX : float
        Horizontal scaling factor (>0)

    scaleY : float
        Vertical scaling factor (>0)

    method : str
        interpolation method
        options:
            "nearest"
            "bilinear"

    Returns
    -------
    numpy.ndarray
        Resized image with shape:

            new_height = old_height * scaleY
            new_width  = old_width  * scaleX

    Raises
    ------
    TypeError
        If image is not numpy array

    ValueError
        If method unsupported or scale ≤ 0
        
    Notes
    -----
        Output image dtype preserved as float32.    
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")

    if method not in ["nearest", "bilinear"]:
        raise ValueError("method must be nearest or bilinear")

    oldHeight, oldWidth = image.shape[:2]

    new_width = int(oldWidth * scaleX)
    new_height = int(oldHeight * scaleY)

    # grayscale image
    if image.ndim == 2:

        output = np.zeros((new_height, new_width))

        for y_new in range(new_height):
            for x_new in range(new_width):

                x_old = x_new * (oldWidth / new_width)
                y_old = y_new * (oldHeight / new_height)

                if method == "nearest":

                    output[y_new, x_new] = nearest_neighbor_interpolation(
                        image,
                        x_old,
                        y_old
                    )

                else:

                    output[y_new, x_new] = bilinear_interpolation(
                        image,
                        x_old,
                        y_old,
                        2
                    )

    # RGB image
    elif image.ndim == 3:

        channels = image.shape[2]

        output = np.zeros((new_height, new_width, channels))

        for c in range(channels):

            for y_new in range(new_height):

                for x_new in range(new_width):

                    x_old = x_new * (oldWidth / new_width)
                    y_old = y_new * (oldHeight / new_height)

                    if method == "nearest":

                        output[y_new, x_new, c] = nearest_neighbor_interpolation(
                            image[:, :, c],
                            x_old,
                            y_old
                        )

                    else:

                        output[y_new, x_new, c] = bilinear_interpolation(
                            image[:, :, c],
                            x_old,
                            y_old,
                            2
                        )

    return output


def nearest_neighbor_interpolation(image, x, y):
    """
    Compute interpolated pixel using nearest neighbor rule.

    Nearest neighbor selects the closest pixel:

        x_nn = round(x)
        y_nn = round(y)

    Result:

        I(x',y') = I(round(x), round(y))

    Characteristics:

        very fast
        preserves edges
        produces blocky appearance when enlarging

    Parameters
    ----------
    image : numpy.ndarray
        single grayscale channel

    x : float
        horizontal coordinate

    y : float
        vertical coordinate

    Returns
    -------
    float
        interpolated pixel intensity
    """

    x_nn = int(round(x))
    y_nn = int(round(y))

    x_nn = max(0, min(x_nn, image.shape[1] - 1))
    y_nn = max(0, min(y_nn, image.shape[0] - 1))

    return image[y_nn, x_nn]


def translate_image(image, tx, ty):
    """
    Translate (shift) image in spatial domain.

    Translation model:

        x' = x + tx
        y' = y + ty

    Implemented using inverse mapping:

        x_old = x_new - tx
        y_old = y_new - ty

    Inverse mapping ensures no holes appear in output image.

    Bilinear interpolation used for subpixel accuracy.

    Parameters
    ----------
    image : numpy.ndarray
        input grayscale or RGB image

    tx : float
        horizontal shift in pixels
        positive → shift right

    ty : float
        vertical shift in pixels
        positive → shift down

    Returns
    -------
    numpy.ndarray
        translated image (same size as input)

    Raises
    ------
    TypeError
        invalid parameter types

    ValueError
        invalid image dimensions
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    height, width = image.shape[:2]

    pad_size = max(abs(int(tx)), abs(int(ty))) + 1

    if image.ndim == 2:

        output = np.zeros((height, width))

        for y_new in range(height):
            for x_new in range(width):

                x_old = x_new - tx
                y_old = y_new - ty

                output[y_new, x_new] = bilinear_interpolation(
                    image,
                    x_old,
                    y_old,
                    pad_size
                )

    elif image.ndim == 3:

        channels = image.shape[2]

        output = np.zeros((height, width, channels))

        for c in range(channels):

            for y_new in range(height):
                for x_new in range(width):

                    x_old = x_new - tx
                    y_old = y_new - ty

                    output[y_new, x_new, c] = bilinear_interpolation(
                        image[:, :, c],
                        x_old,
                        y_old,
                        pad_size
                    )

    return output


def rotate_image(image, angle_degrees):
    """
    Rotate image about its center using inverse mapping.

    Rotation matrix:

        [x']   [ cosθ  -sinθ ][x]
        [y'] = [ sinθ   cosθ ][y]

    inverse mapping used:

        [x]   [ cosθ   sinθ ][x']
        [y] = [-sinθ   cosθ ][y']

    Steps:

        1. shift origin to image center
        2. apply inverse rotation
        3. interpolate intensity value

    Bilinear interpolation ensures smooth rotation.

    Output image size expanded to avoid cropping.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale or RGB image

    angle_degrees : float
        rotation angle in degrees

    Returns
    -------
    numpy.ndarray
        rotated image

    Raises
    ------
    TypeError
        invalid parameter types

    ValueError
        invalid image shape
        
    Notes
    -----
        Output image size automatically expanded
        to avoid information loss.    
    """

    angle_radians = np.radians(angle_degrees)

    old_height, old_width = image.shape[:2]

    new_width = int(
        abs(old_width * np.cos(angle_radians)) +
        abs(old_height * np.sin(angle_radians))
    )

    new_height = int(
        abs(old_height * np.cos(angle_radians)) +
        abs(old_width * np.sin(angle_radians))
    )

    old_cx = old_width / 2
    old_cy = old_height / 2

    new_cx = new_width / 2
    new_cy = new_height / 2

    pad_size = int(max(old_width, old_height)) + 2

    if image.ndim == 2:

        output = np.zeros((new_height, new_width))

        for y_new in range(new_height):
            for x_new in range(new_width):

                x_shifted = x_new - new_cx
                y_shifted = y_new - new_cy

                x_old = (
                    x_shifted * np.cos(angle_radians) +
                    y_shifted * np.sin(angle_radians)
                )

                y_old = (
                    -x_shifted * np.sin(angle_radians) +
                    y_shifted * np.cos(angle_radians)
                )

                x_old += old_cx
                y_old += old_cy

                output[y_new, x_new] = bilinear_interpolation(
                    image,
                    x_old,
                    y_old,
                    pad_size
                )

    elif image.ndim == 3:

        channels = image.shape[2]

        output = np.zeros((new_height, new_width, channels))

        for c in range(channels):

            for y_new in range(new_height):
                for x_new in range(new_width):

                    x_shifted = x_new - new_cx
                    y_shifted = y_new - new_cy

                    x_old = (
                        x_shifted * np.cos(angle_radians) +
                        y_shifted * np.sin(angle_radians)
                    )

                    y_old = (
                        -x_shifted * np.sin(angle_radians) +
                        y_shifted * np.cos(angle_radians)
                    )

                    x_old += old_cx
                    y_old += old_cy

                    output[y_new, x_new, c] = bilinear_interpolation(
                        image[:, :, c],
                        x_old,
                        y_old,
                        pad_size
                    )

    return output


def bilinear_interpolation(image, x, y, pad_size):
    """
    Compute interpolated pixel using bilinear interpolation.

    Bilinear interpolation uses weighted average
    of four nearest pixels:

        Q11, Q21
        Q12, Q22

    interpolation steps:

        top    = Q11(1-dx) + Q21(dx)
        bottom = Q12(1-dx) + Q22(dx)

        value  = top(1-dy) + bottom(dy)

    ensures smooth intensity transitions.

    Parameters
    ----------
    image : numpy.ndarray
        single grayscale channel

    x : float
        horizontal coordinate

    y : float
        vertical coordinate

    pad_size : int
        padding to avoid boundary errors

    Returns
    -------
    float
        interpolated pixel value
    """

    padded_image = fdf.pad_image(image, pad_size)

    x = x + pad_size
    y = y + pad_size

    x1 = int(np.floor(x))
    y1 = int(np.floor(y))

    x2 = x1 + 1
    y2 = y1 + 1

    x1 = max(0, min(x1, padded_image.shape[1] - 1))
    x2 = max(0, min(x2, padded_image.shape[1] - 1))
    y1 = max(0, min(y1, padded_image.shape[0] - 1))
    y2 = max(0, min(y2, padded_image.shape[0] - 1))

    dx = x - x1
    dy = y - y1

    q11 = padded_image[y1, x1]
    q21 = padded_image[y1, x2]
    q12 = padded_image[y2, x1]
    q22 = padded_image[y2, x2]

    top = q11 * (1 - dx) + q21 * dx
    bottom = q12 * (1 - dx) + q22 * dx

    value = top * (1 - dy) + bottom * dy

    return value