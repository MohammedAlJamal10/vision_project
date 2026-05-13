import numpy as np
import filters as filt
import utils as fdf



"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""


# -----------------------------------
# Global Descriptor 1: Mean Intensity
# -----------------------------------
def mean_intensity_descriptor(image):
    """
    Compute average intensity of grayscale image.

    Mathematical definition:

        μ = (1/N) Σ I(x,y)

    Captures overall brightness of image.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image

    Returns
    -------
    float
        mean pixel intensity
    """

    if image.ndim != 2:
        raise ValueError("grayscale required")

    return np.mean(image)


# -----------------------------------
# Global Descriptor 2: Histogram Descriptor
# -----------------------------------
def histogram_descriptor(image, bins=16):
    """
    Compute normalized intensity histogram.

    Image automatically converted to 8-bit range.

    Histogram definition:

        h(r_k) = number of pixels
        with intensity inside bin k

    descriptor length = bins

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image

    bins : int
        number of bins

    Returns
    -------
    numpy.ndarray
        normalized histogram vector
    """

    if image.ndim != 2:
        raise ValueError("grayscale required")


    img = fdf.normalize_image(
        image,
        "0to255"
    ).astype(np.uint8)


    hist = np.zeros(bins)

    bin_width = 256/bins


    for val in img.flatten():

        idx = int(val//bin_width)

        if idx >= bins:
            idx = bins-1

        hist[idx]+=1


    hist = hist/np.sum(hist)

    return hist


# -----------------------------------
# Gradient Descriptor 1: Edge Density
# -----------------------------------
def edge_density_descriptor(image, threshold=0.2):
    """
    Compute proportion of edge pixels.

    Sobel gradient magnitude:

        G = sqrt(Gx^2 + Gy^2)

    Edge density:

        edge_pixels / total_pixels

    threshold relative to normalized magnitude.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image

    threshold : float
        threshold in normalized range

    Returns
    -------
    float
        edge density feature
    """

    gx,gy,mag = filt.sobel_gradients(image)

    mag = fdf.normalize_image(mag,"0to1")

    edges = mag>threshold

    density = np.sum(edges)/image.size

    return density


# -----------------------------------
# Gradient Descriptor 2: Orientation Histogram
# -----------------------------------
def gradient_orientation_descriptor(image,bins=8):
    """
    Histogram of gradient orientations.

    orientation computed using:

        θ = arctan(Gy/Gx)

    angle range:

        0 → 180 degrees

    descriptor length = bins

    weighted by gradient magnitude.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image

    bins : int
        number of angle bins

    Returns
    -------
    numpy.ndarray
        normalized orientation histogram
    """

    gx,gy,mag = filt.sobel_gradients(image)

    angles = np.arctan2(gy,gx)*180/np.pi

    angles[angles<0]+=180


    hist = np.zeros(bins)

    bin_width = 180/bins


    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            idx = int(angles[i,j]//bin_width)

            if idx>=bins:
                idx=bins-1

            hist[idx]+=mag[i,j]


    hist = hist/(np.sum(hist)+1e-8)

    return hist


# -----------------------------------
# Combined Feature Vector
# -----------------------------------
def extract_feature_vector(image):
    """
    Combine all descriptors into single vector.

    final feature dimension:

        1 mean intensity
        16 histogram bins
        1 edge density
        8 orientation bins

    total dimension = 26 features

    Parameters
    ----------
    image : numpy.ndarray
        grayscale image

    Returns
    -------
    np.ndarray

    feature vector

    dimension = 26

        1 mean intensity
        16 histogram bins
        1 edge density
        8 gradient orientation bins
    """

    mean_feat = np.array(
        [mean_intensity_descriptor(image)]
    )

    hist_feat = histogram_descriptor(
        image,
        16
    )

    edge_feat = np.array(
        [edge_density_descriptor(image)]
    )

    orient_feat = gradient_orientation_descriptor(
        image,
        8
    )

    return np.concatenate([
        mean_feat,
        hist_feat,
        edge_feat,
        orient_feat
    ])