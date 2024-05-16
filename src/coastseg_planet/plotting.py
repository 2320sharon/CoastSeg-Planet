import rasterio
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image
import numpy as np

def plot_overlay(segfile, bigimage, color_label, N_DATA_BANDS):
    """
    Plot an overlay of a color label on top of a big image.

    Parameters:
    - segfile (str): The file path to save the overlay image.
    - bigimage (numpy.ndarray): The big image to be displayed as the background.
    - color_label (numpy.ndarray): The color label to be overlaid on top of the big image.
    - N_DATA_BANDS (int): The number of data bands in the big image.

    Returns:
    None
    """
    plt.imshow(bigimage, cmap='gray' if N_DATA_BANDS <= 3 else None)
    plt.imshow(color_label, alpha=0.5)
    plt.axis("off")
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

def plot_side_by_side_overlay(segfile, bigimage, color_label, N_DATA_BANDS):
    """
    Plot two images side by side with an overlay of a color label.

    Parameters:
    - segfile (str): The file path to save the resulting image.
    - bigimage (numpy.ndarray): The image to be displayed on the left side.
    - color_label (numpy.ndarray): The color label to be overlaid on the right side.
    - N_DATA_BANDS (int): The number of data bands in the image.

    Returns:
    None
    """
    plt.subplot(121)
    plt.imshow(bigimage, cmap='gray' if N_DATA_BANDS <= 3 else None)
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(bigimage, cmap='gray' if N_DATA_BANDS <= 3 else None)
    plt.imshow(color_label, alpha=0.5)
    plt.axis("off")
    plt.savefig(segfile, dpi=200, bbox_inches="tight")
    plt.close("all")

def plot_per_class_probabilities(segfile, bigimage, softmax_scores, N_DATA_BANDS):
    """
    Plot per-class probabilities on top of the input image.

    Args:
        segfile (str): The path to the segmentation file.
        bigimage (numpy.ndarray): The input image.
        softmax_scores (numpy.ndarray): The softmax scores for each class.
        N_DATA_BANDS (int): The number of data bands in the image.

    Returns:
        None
    """
    for kclass in range(softmax_scores.shape[-1]):
        tmpfile = segfile.replace("_overlay.png", "_overlay_"+str(kclass)+"prob.png")
        plt.imshow(bigimage, cmap='gray' if N_DATA_BANDS <= 3 else None)
        plt.imshow(softmax_scores[:,:,kclass], alpha=0.5, vmax=1, vmin=0)
        plt.axis("off")
        plt.colorbar()
        plt.savefig(tmpfile, dpi=200, bbox_inches="tight")
        plt.close("all")

def visualize_raster(REFERENCE):
    """
    Visualizes a raster image.

    Parameters:
    - REFERENCE (str): The file path to the raster image.

    Returns:
    - None

    """
    with rasterio.open(REFERENCE) as src:
        # Read the raster data
        data = src.read()

        # Visualize the raster data
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(data[i], cmap='gray')
            ax.set_title(f'Band {i+1}')
            ax.axis('off')
        plt.show()
