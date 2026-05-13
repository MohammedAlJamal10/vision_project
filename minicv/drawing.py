import numpy as np
import image_io as io
import utils as fdf
import transforms as gt
import filters as filt
import features as feat


"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
image expected in range [0,1]
"""


# -----------------------------------
# Helper Function
# -----------------------------------
def set_pixel(image, x, y, color):
    """
    Safely assign pixel value if coordinates lie inside image boundaries.

    Supports both grayscale and RGB images.

    Boundary handling:
        pixels outside image limits are ignored (clipped).

    Color format:

        grayscale → single numeric value
        RGB → tuple/list of 3 values

    Examples:

        set_pixel(img,10,20,1.0)
        set_pixel(img,10,20,(1,0,0))

    Parameters
    ----------
    image : numpy.ndarray
        grayscale (H,W) or RGB (H,W,3)

    x : int
        horizontal coordinate

    y : int
        vertical coordinate

    color : float | tuple
        pixel intensity or RGB value

    Returns
    -------
    None
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if not isinstance(x, int):
        raise TypeError("x must be an integer.")

    if not isinstance(y, int):
        raise TypeError("y must be an integer.")

    if image.ndim not in [2, 3]:
        raise ValueError("Image must be grayscale (2D) or RGB (3D).")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("RGB image must have exactly 3 channels.")

    # Validate grayscale color
    if image.ndim == 2:
        if not isinstance(color, (int, float)):
            raise TypeError("For grayscale images, color must be numeric.")

    # Validate RGB color
    elif image.ndim == 3:
        if not isinstance(color, (tuple, list, np.ndarray)):
            raise TypeError("For RGB images, color must be a tuple, list, or NumPy array.")

        if len(color) != 3:
            raise ValueError("RGB color must contain exactly 3 values.")

    height, width = image.shape[:2]

    if 0 <= x < width and 0 <= y < height:

        if image.ndim == 2:
            image[y, x] = color

        elif image.ndim == 3:
            image[y, x] = color


# -----------------------------------
# Draw Point
# -----------------------------------
def draw_point(image, x, y, color=1.0, thickness=1):
    """
    Draw point centered at (x,y).

    Thickness implemented as square neighborhood:

        thickness = 1 → single pixel
        thickness = 3 → 3x3 square

    Works for normalized images:

        grayscale range → [0,1]
        RGB range → [0,1]

    Parameters
    ----------
    image : numpy.ndarray

    x,y : int
        pixel coordinates

    color : float or tuple
        pixel value

    thickness : int
        size of point region

    Returns
    -------
    numpy.ndarray
        modified image
    """

    if not isinstance(thickness, int):
        raise TypeError("thickness must be an integer.")

    if thickness < 1:
        raise ValueError("thickness must be at least 1.")

    radius = thickness // 2

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            set_pixel(image, x + dx, y + dy, color)

    return image


# -----------------------------------
# Draw Line using Bresenham
# -----------------------------------
def draw_line(image, x1, y1, x2, y2, color=1.0, thickness=1):
    """
    Draw straight line using Bresenham algorithm.

    Bresenham method efficiently determines
    which pixels best approximate a line:

        minimizes floating point operations
        ensures connectivity of pixels

    Thickness handled by drawing multiple
    parallel points around each line pixel.

    Parameters
    ----------
    image : numpy.ndarray

    x1,y1 : int
        start coordinate

    x2,y2 : int
        end coordinate

    color : float | tuple

    thickness : int

    Returns
    -------
    numpy.ndarray
    """

    for value, name in zip([x1, y1, x2, y2], ["x1", "y1", "x2", "y2"]):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

    if not isinstance(thickness, int):
        raise TypeError("thickness must be an integer.")

    if thickness < 1:
        raise ValueError("thickness must be at least 1.")

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    while True:
        draw_point(image, x1, y1, color, thickness)

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy

    return image


# -----------------------------------
# Draw Rectangle
# -----------------------------------
def draw_rectangle(image, x1, y1, x2, y2, color=1.0, thickness=1, filled=False):
    """
    Draw rectangle defined by two opposite corners.

    Outline implemented using 4 lines.

    Filled rectangle uses raster filling.

    Coordinates:

        (x1,y1) → top-left
        (x2,y2) → bottom-right

    Parameters
    ----------
    image : numpy.ndarray

    color : float | tuple

    thickness : int
        border width

    filled : bool
        True → filled rectangle

    Returns
    -------
    numpy.ndarray
    """

    for value, name in zip([x1, y1, x2, y2], ["x1", "y1", "x2", "y2"]):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

    if not isinstance(thickness, int):
        raise TypeError("thickness must be an integer.")

    if thickness < 1:
        raise ValueError("thickness must be at least 1.")

    if not isinstance(filled, bool):
        raise TypeError("filled must be a boolean.")

    if filled:
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                set_pixel(image, x, y, color)

    else:
        draw_line(image, x1, y1, x2, y1, color, thickness)
        draw_line(image, x2, y1, x2, y2, color, thickness)
        draw_line(image, x2, y2, x1, y2, color, thickness)
        draw_line(image, x1, y2, x1, y1, color, thickness)

    return image


# -----------------------------------
# Draw Polygon
# -----------------------------------
def draw_polygon(image, points, color=1.0, thickness=1):
    """
    Draw polygon edges connecting sequence of points.

    Polygon automatically closed:

        last point connected to first point.

    Works for any number of vertices >=2.

    Example:

        triangle → 3 points
        pentagon → 5 points

    Parameters
    ----------
    image : numpy.ndarray

    points : list[(x,y)]

    color : float | tuple

    thickness : int

    Returns
    -------
    numpy.ndarray
    """

    if not isinstance(points, list):
        raise TypeError("points must be a list of (x, y) tuples.")

    if len(points) < 2:
        raise ValueError("Polygon must contain at least 2 points.")

    for point in points:
        if not isinstance(point, tuple):
            raise TypeError("Each point must be a tuple.")

        if len(point) != 2:
            raise ValueError("Each point must contain exactly 2 values.")

        if not all(isinstance(coord, int) for coord in point):
            raise TypeError("Point coordinates must be integers.")

    if not isinstance(thickness, int):
        raise TypeError("thickness must be an integer.")

    if thickness < 1:
        raise ValueError("thickness must be at least 1.")

    num_points = len(points)

    for i in range(num_points):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % num_points]

        draw_line(image, x1, y1, x2, y2, color, thickness)

    return image