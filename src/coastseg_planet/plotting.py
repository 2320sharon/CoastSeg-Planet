import rasterio
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import colorsys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

def create_class_color_mapping(mask):
    """
    Create a color mapping for each class in the segmentation mask.
    
    Parameters:
    - mask: numpy array of the segmentation mask (H x W).
    
    Returns:
    - class_color_mapping: dictionary with class values as keys and colors as values.
    """
    unique_classes = np.unique(mask)
    colors = plt.cm.get_cmap('jet', len(unique_classes))
    class_color_mapping = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    return class_color_mapping

def create_color_mapping(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. 
    The colors are generated using the HLS color model, and the resulting RGB values are floating-point numbers in the range of 0.0-1.0.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of floating-point numbers.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [
            x for x in colorsys.hls_to_rgb(h, 0.5, 1.0)
        ] 
        color_mapping[num] = (r, g, b)

    print(f"color_mapping: {color_mapping}")
    return color_mapping

def create_classes_overlay_image(labels):
    """
    Creates an overlay image by mapping class labels to colors.

    Args:
    labels (numpy.ndarray): A 2D array representing class labels for each pixel in an image.

    Returns:
    numpy.ndarray: A 3D array representing an overlay image with the same size as the input labels.
    """
    # Ensure that the input labels is a NumPy array
    labels = np.asarray(labels)
    print(f"labels: {labels}")
    print(f"labels shape: {labels.shape}")

    # Make an overlay the same size of the image with 3 color channels
    overlay_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)

    # Create a color mapping for the labels
    class_indices = np.unique(labels)
    color_mapping = create_color_mapping(class_indices)
    print(f"color_mapping: {color_mapping}")

    # Create the overlay image by assigning the color for each label
    for index, class_color in color_mapping.items():
        print(f"index: {index}, class_color: {class_color}")
        mask = (labels == index)
        overlay_image[mask] = class_color
        plt.imshow(overlay_image)
        print(os.path.abspath(f"overlay_image{index}.png"))
        plt.savefig(f"overlay_image{index}.png")
        print(f"Updated overlay_image for index {index}: {overlay_image[mask]}")

    # # Create the overlay image by assigning the color for each label
    # for index, class_color in color_mapping.items():
    #     print(f"index: {index}, class_color: {class_color}")
    #     print(f"overlay_image[labels == index]: {np.any(overlay_image[labels == index])}")
    #     print(f"overlay_image[labels == index]: {overlay_image[labels == index]}")
    #     overlay_image[index] = class_color

    return overlay_image

def create_overlay(
    im_RGB: "np.ndarray[float]",
    im_labels: "np.ndarray[int]",
    overlay_opacity: float = 0.35,
) -> "np.ndarray[float]":
    """
    Create an overlay on the given image using the provided labels and
    specified overlay opacity.

    Args:
    im_RGB (np.ndarray[float]): The input image as an RGB numpy array (height, width, 3).
    im_labels (np.ndarray[int]): The array containing integer labels of the same dimensions as the input image.
    overlay_opacity (float, optional): The opacity value for the overlay (default: 0.35).

    Returns:
    np.ndarray[float]: The combined numpy array of the input image and the overlay.
    """
    # Create an overlay using the given labels
    print(f"im_labels : {np.unique(im_labels)}")
    overlay = create_classes_overlay_image(im_labels)
    print(f"overlay shape: {overlay.shape}")
    print(f"overlay values: {np.unique(overlay)}")
    # return overlay
    # Combine the original image and the overlay using the correct opacity
    combined_float = im_RGB * (1 - overlay_opacity) + overlay * overlay_opacity
    return combined_float

def save_detection_figure(fig, filepath: str, date: str, satname: str) -> None:
    """
    Save the given figure as a jpg file with a specified dpi.

    Args:
    fig (Figure): The figure object to save.
    filepath (str): The directory path where the image will be saved.
    date (str): The date the satellite image was taken in the format 'YYYYMMDD'.
    satname (str): The name of the satellite that took the image.

    Returns:
    None
    """
    print(f"Saving figure to {os.path.join(filepath, date + '_' + satname + '.jpg')}")
    fig.savefig(
        os.path.join(filepath, date + "_" + satname + ".jpg"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close the figure after saving
    plt.close("all")
    del fig

def create_color_mapping_as_ints(int_list: list[int]) -> dict:
    """
    This function creates a color mapping dictionary for a given list of integers, assigning a unique RGB color to each integer. The colors are generated using the HLS color model, and the resulting RGB values are integers in the range of 0-255.

    Arguments:

    int_list (list): A list of integers for which unique colors need to be generated.
    Returns:

    color_mapping (dict): A dictionary where the keys are the input integers and the values are the corresponding RGB colors as tuples of integers.
    """
    n = len(int_list)
    h_step = 1.0 / n
    color_mapping = {}

    for i, num in enumerate(int_list):
        h = i * h_step
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(h, 0.5, 1.0)]
        color_mapping[num] = (r, g, b)

    return color_mapping

def create_legend(
    class_mapping: dict, color_mapping: dict = None, additional_patches: list = None
) -> list[mpatches.Patch]:
    """
    Creates a list of legend patches using class and color mappings.

    Args:
    class_mapping (dict): A dictionary mapping class indices to class names.
    color_mapping (dict, optional): A dictionary mapping class indices to colors. Defaults to None.
    additional_patches (list, optional): A list of additional patches to be appended to the legend. Defaults to None.

    Returns:
    list: A list of legend patches.
    """

    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
    #            for color in class_color_mapping.values()]
    # labels = [class_name_dict[cls] for cls in class_color_mapping.keys()]
    # plt.legend(handles, labels, title='Classes', loc='lower right')

    legend = [
        mpatches.Patch(
            color=color, label=f"{class_mapping.get(index, f'{index}')}"
        )
        for index, color in color_mapping.items()
    ]

    return legend + additional_patches if additional_patches else legend

def create_mask_image(mask, color_mapping):
    """
    Create a colored mask image based on the given mask and color mapping.
    
    Creates a 4 channel images RGB adn Alpha channels.
    Parameters:
    - mask (ndarray): The mask array representing the segmentation mask.
    - color_mapping (dict): A dictionary mapping class labels to RGBA color values.
    
    Returns:
    - colored_mask (ndarray): The colored mask image with RGBA channels.
    Example:
    color_mapping = {0: [255, 0, 0, 255], 1: [0, 255, 0, 255], 2: [0, 0, 255, 255]}
    mask = np.array([[0, 1, 2], [1, 2, 0]])
    colored_mask = create_mask_image(mask, color_mapping)
    """
    colored_mask = np.zeros((*mask.shape, 4))
    for cls, color in color_mapping.items():
        colored_mask[mask == cls] = color
    return colored_mask


def plot_image_with_legend(
    original_image: "np.ndarray[float]",
    land_water_mask: "np.ndarray[float]",
    all_mask: "np.ndarray[float]",
    pixelated_shoreline: "np.ndarray[float]",
    merged_legend,
    all_legend,
    class_color_mapping: dict,
    all_class_color_mapping:dict,
    titles: list[str] = [],
    overlay_opacity: float=0.35,
):

    if not titles or len(titles) != 3:
        titles = ["Original Image", "Merged Classes", "All Classes"]
    fig = plt.figure()
    fig.set_size_inches([18, 9])

    if original_image.shape[1] > 2.5 * original_image.shape[0]:
        gs = gridspec.GridSpec(3, 1)
    else:
        gs = gridspec.GridSpec(1, 3)

    # if original_image is wider than 2.5 times as tall, plot the images in a 3x1 grid (vertical)
    if original_image.shape[0] > 2.5 * original_image.shape[1]:
        # vertical layout 3x1
        gs = gridspec.GridSpec(3, 1)
        ax2_idx, ax3_idx = (1, 0), (2, 0)
        bbox_to_anchor = (1.05, 0.5)
        loc = "center left"
    else:
        # horizontal layout 1x3
        gs = gridspec.GridSpec(1, 3)
        ax2_idx, ax3_idx = (0, 1), (0, 2)
        bbox_to_anchor = (0.5, -0.23)
        loc = "lower center"

    # bbox_to_anchor = (0.5, -0.1)  # Move the legend below the image

    gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[ax2_idx], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[ax3_idx], sharex=ax1, sharey=ax1)

    # Plot original image
    ax1.imshow(original_image)
    # ax1.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    ax1.set_title(titles[0])
    ax1.axis("off")


    # colored_mask = np.zeros((*land_water_mask.shape, 4))
    # for cls, color in class_color_mapping.items():
    #     colored_mask[land_water_mask == cls] = color
    # Create merged mask and plot
    merged_mask = create_mask_image(land_water_mask, class_color_mapping)
    # # Plot the second image that has the merged the water classes and all the land classes together
    ax2.imshow(original_image)
    ax2.imshow(merged_mask, alpha=overlay_opacity)
    ax2.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    ax2.set_title(titles[1])
    ax2.axis("off")
    

    ax2.legend(
        handles=merged_legend,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
        borderaxespad=0.0,
        fontsize='small',  # Reduce legend font size
        markerscale=0.7  # Reduce marker size
    )
    
    # # # Create dynamic legend handles and labels
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
    #            for color in class_color_mapping.values()]
    # labels = [class_name_dict[cls] for cls in class_color_mapping.keys()]
    # ax2.legend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=bbox_to_anchor,
    #     loc=loc,
    #     borderaxespad=0.0,
    # )
    
    # colored_mask = np.zeros((*all_mask.shape, 4))
    # for cls, color in all_class_color_mapping.items():
    #     colored_mask[all_mask == cls] = color

    all_labels_mask = create_mask_image(all_mask, all_class_color_mapping)
    # # Plot the second image that has the merged the water classes and all the land classes together
    ax3.imshow(original_image)    
    ax3.imshow(all_labels_mask, alpha=overlay_opacity)
    ax3.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
    ax3.set_title(titles[2])
    ax3.axis("off")
    if all_legend:  # Check if the list is not empty
        ax3.legend(
            handles=all_legend,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            borderaxespad=0.0,
            fontsize='small',  # Reduce legend font size
            markerscale=0.7  # Reduce marker size
        )

    # Return the figure object
    return fig


# def plot_image_with_legend(
#     original_image: "np.ndarray[float]",
#     merged_overlay: "np.ndarray[float]",
#     all_overlay: "np.ndarray[float]",
#     pixelated_shoreline: "np.ndarray[float]",
#     merged_legend: list,
#     all_legend: list,
#     titles: list[str] = [],
# ):

#     if not titles or len(titles) != 3:
#         titles = ["Original Image", "Merged Classes", "All Classes"]
#     fig = plt.figure()
#     fig.set_size_inches([18, 9])

#     if original_image.shape[1] > 2.5 * original_image.shape[0]:
#         gs = gridspec.GridSpec(3, 1)
#     else:
#         gs = gridspec.GridSpec(1, 3)

#     # if original_image is wider than 2.5 times as tall, plot the images in a 3x1 grid (vertical)
#     if original_image.shape[0] > 2.5 * original_image.shape[1]:
#         # vertical layout 3x1
#         gs = gridspec.GridSpec(3, 1)
#         ax2_idx, ax3_idx = (1, 0), (2, 0)
#         bbox_to_anchor = (1.05, 0.5)
#         loc = "center left"
#     else:
#         # horizontal layout 1x3
#         gs = gridspec.GridSpec(1, 3)
#         ax2_idx, ax3_idx = (0, 1), (0, 2)
#         bbox_to_anchor = (0.5, -0.23)
#         loc = "lower center"

#     gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[ax2_idx], sharex=ax1, sharey=ax1)
#     ax3 = fig.add_subplot(gs[ax3_idx], sharex=ax1, sharey=ax1)

#     # Plot original image
#     ax1.imshow(original_image)
#     ax1.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
#     ax1.set_title(titles[0])
#     ax1.axis("off")

#     # Plot the second image that has the merged the water classes and all the land classes together
#     ax2.imshow(merged_overlay)
#     ax2.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
#     ax2.set_title(titles[1])
#     ax2.axis("off")
#     if merged_legend:  # Check if the list is not empty
#         ax2.legend(
#             handles=merged_legend,
#             bbox_to_anchor=bbox_to_anchor,
#             loc=loc,
#             borderaxespad=0.0,
#         )

#     # Plot the second image that shows all the classes separately
#     ax3.imshow(all_overlay)
#     ax3.plot(pixelated_shoreline[:, 0], pixelated_shoreline[:, 1], "k.", markersize=1)
#     ax3.set_title(titles[2])
#     ax3.axis("off")
#     if all_legend:  # Check if the list is not empty
#         ax3.legend(
#             handles=all_legend,
#             bbox_to_anchor=bbox_to_anchor,
#             loc=loc,
#             borderaxespad=0.0,
#         )

#     # Return the figure object
#     return fig


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
