import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg



"""
MiniCV convention:

image dtype  : float32
pixel range  : [0,1]
grayscale    : shape (H,W)
RGB          : shape (H,W,3)
"""



def put_text(image, text, x, y, font_size=20, color=(1,1,1)):
    """
    Draw text onto grayscale or RGB image using matplotlib rendering.

    Supports normalized MiniCV format:

        grayscale range → [0,1]
        RGB range → [0,1]

    Text rendered using matplotlib canvas and converted back to numpy array.

    Parameters
    ----------
    image : numpy.ndarray
        grayscale (H,W) or RGB (H,W,3)

    text : str
        text string to render

    x : int or float
        horizontal text position

    y : int or float
        vertical text position

    font_size : int or float
        text size

    color : float | tuple | str
        text color

        grayscale examples:
            1.0 → white
            0.0 → black

        RGB examples:
            (1,0,0) → red
            (0,1,0) → green

        matplotlib color strings also allowed:
            "white"
            "red"

    Returns
    -------
    numpy.ndarray
        image with text drawn (same shape as input)

    Raises
    ------
    TypeError
        invalid input types

    ValueError
        invalid image shape or color format
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy array")

    if image.ndim not in [2,3]:
        raise ValueError("image must be 2D or 3D")

    if image.ndim==3 and image.shape[2]!=3:
        raise ValueError("RGB image must have 3 channels")

    if not isinstance(text,str):
        raise TypeError("text must be string")

    height,width = image.shape[:2]


    fig,ax = plt.subplots(
        figsize=(width/100,height/100),
        dpi=100
    )

    ax.set_axis_off()
    fig.subplots_adjust(0,0,1,1)


    # show image
    if image.ndim==2:
        ax.imshow(image,cmap="gray",vmin=0,vmax=1)

    else:
        ax.imshow(image)


    # handle grayscale color
    if image.ndim==2 and isinstance(color,(int,float)):

        if not 0<=color<=1:
            raise ValueError("grayscale color must be in [0,1]")

        color=str(color)


    # handle RGB color
    if image.ndim==3 and isinstance(color,(tuple,list)):

        if len(color)!=3:
            raise ValueError("RGB must have 3 values")

        for c in color:
            if not 0<=c<=1:
                raise ValueError("RGB values must be in [0,1]")


    ax.text(
        x,
        y,
        text,
        fontsize=font_size,
        color=color
    )


    canvas = FigureCanvasAgg(fig)

    canvas.draw()

    rendered = np.asarray(
        canvas.buffer_rgba()
    )[:,:,:3]


    plt.close(fig)


    rendered = rendered.astype(np.float32)/255


    # convert back to grayscale if needed
    if image.ndim==2:

        rendered = (
            0.299*rendered[:,:,0] +
            0.587*rendered[:,:,1] +
            0.114*rendered[:,:,2]
        )


    return rendered