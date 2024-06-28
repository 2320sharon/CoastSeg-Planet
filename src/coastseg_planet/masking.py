import os
import rasterio
import numpy as np
import glob
from skimage import morphology
from coastseg_planet.shoreline_extraction import get_class_indices_from_model_card
from coastseg_planet.processing import get_base_filename

def create_water_bad_mask_directory(directory: str,
                                    model_card_path: str,
                                    input_suffix: str = "*udm2_clip_combined_mask.tif",
                                    output_suffix: str = "_bad_mask.tif",
                                    npz_suffix: str = "_TOAR_processed_coregistered_global_res.npz",
                                    separator="_3B",
                                    ):
    """
    Creates combines the water masks from each segmentation with the bad mask (no data & cloud mask) and saves them to a new file with the given suffix for a given directory of cloud paths.

    Args:
        directory (str): The directory containing the cloud paths.
        model_card_path (str): The path to the model card.
        input_suffix (str, optional): The suffix of the input files. Defaults to "*udm2_clip_combined_mask.tif".
        output_suffix (str, optional): The suffix of the output files. Defaults to "_bad_mask.tif".
        npz_suffix (str, optional): The suffix of the npz files. Defaults to "_TOAR_processed_coregistered_global_res.npz".
        separator (str, optional): The separator used in the filenames. Defaults to "_3B".

    Returns:
        None
    """
    cloud_paths = glob.glob(os.path.join(directory, f"*{separator}{input_suffix}"))
    for cloud_path in cloud_paths:
        base_filename = get_base_filename(cloud_path, separator)
        npz_path = os.path.join(directory, 'out', f"{base_filename}{separator}{npz_suffix}")
        if not os.path.isfile(npz_path) or not npz_path.endswith(".npz"):
            print(f"Skipping {npz_path} because it does not exist")
            continue
        water_classes_indices, water_mask, class_mapping  = get_class_indices_from_model_card(npz_path,model_card_path) 
        output_path = combine_masks(cloud_path,water_mask,output_suffix)
        print(f"Created {output_path}")

def combine_masks(mask_path,mask:np.ndarray,suffix:str="_bad_mask.tif"):
    """
    Combine a no data mask  with a water mask and save to a new file.

    Parameters:
    mask_path (str): The path to the UDM file.
    mask (numpy.ndarray): The mask to be combined with the bad mask. (typically a water mask for coreg purposes)
    suffix (str): The suffix to be added to the combined mask file.

    Returns:
    combined_path (str): The path to the combined mask file.

    Raises:
    FileNotFoundError: If the UDM file does not exist.
    """
    with rasterio.open(mask_path) as src:
        bad_mask = src.read(1)
        if bad_mask.shape != mask.shape:
            bad_mask = bad_mask[:mask.shape[0],:mask.shape[1]] # cut off the last row and column if the shape does not match

        # combine the cloud and shadow mask
        combined_mask = np.logical_or(bad_mask, mask)

        # write the combined mask to a new file
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

        # write to a new file
        # mask_out = combined_mask > 0
        base_filename = os.path.basename(mask_path.split(".")[-2])
        combined_path = os.path.join(
            os.path.dirname(mask_path), base_filename + suffix
        )

        with rasterio.open(combined_path, "w", **profile) as dst:
            dst.write(combined_mask.astype(rasterio.uint8), 1)

        return combined_path

def apply_cloud_mask_correction(
    im_cloud_mask,
    cloud_values,
):
    """
    Remove cloud pixels that form very thin features. These are beach or swash pixels that are
    erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.

    Parameters:
    - im_QA: numpy array of the image's cloud mask
    - cloud_values: List of values indicating cloud presence.


    Returns:
    - A boolean numpy array where True represents corrected cloud mask presence.
    """
    cloud_mask = np.zeros_like(im_cloud_mask, dtype=bool)
    for value in cloud_values:
        cloud_mask_temp = np.isin(im_cloud_mask, value)
        elem = morphology.square(6)  # use a square of width 6 pixels
        cloud_mask_temp = morphology.binary_opening(
            cloud_mask_temp, elem
        )  # perform image opening
        cloud_mask_temp = morphology.remove_small_objects(
            cloud_mask_temp, min_size=100, connectivity=1
        )
        cloud_mask = np.logical_or(cloud_mask, cloud_mask_temp)
    return cloud_mask


def extract_cloud_flags(cloud_mask):
    """
    Extracts cloud flags from a given cloud mask.
    This works only for landsat 5,7,8 and 9 cloud masks collected from GEE collection 2.

    dilated cloud = bit 1
    cirrus = bit 2
    cloud = bit 3

    Args:
        cloud_mask (numpy.ndarray): The cloud mask array.

    Returns:
        list: A list of cloud values extracted from the cloud mask.

    """

    # function to return flag for n-th bit
    def is_set(x, n):
        return x & 1 << n != 0

    qa_values = np.unique(cloud_mask.flatten())
    cloud_values = []
    for qaval in qa_values:
        for k in [1, 2, 3]:  # check the first 3 flags
            if is_set(qaval, k):
                cloud_values.append(qaval)
    return cloud_values


def process_landsat_cloud_mask(
    ref_mask_path: str, cloud_mask_issue: bool = False
) -> np.ndarray:
    """
    Create a boolean cloud mask from the cloud mask of a Landsat image.

    Works only for Landsat 5, 7, 8 and 9 cloud masks collected from GEE collection 2.

    Args:
        ref_mask_path (str): The file path to the reference cloud mask.
        cloud_mask_issue (bool, optional): Indicates if there is an issue with the cloud mask. Defaults to False.
                Typically when the cloud mask covers the shoreline.

    Returns:
        np.ndarray: The processed boolean cloud mask.

    """
    ref_cloud_mask_array = rasterio.open(ref_mask_path).read()
    # remove the first dimension
    ref_cloud_mask_array = ref_cloud_mask_array[0]
    # find which pixels have bits corresponding to cloud values
    cloud_values = extract_cloud_flags(ref_cloud_mask_array)
    cloud_mask = np.isin(ref_cloud_mask_array, cloud_values)
    if cloud_mask_issue:
        cloud_mask = apply_cloud_mask_correction(cloud_mask, cloud_values)
    return cloud_mask


def save_landsat_cloud_mask(
    ref_mask_path: str, output_path: str, cloud_mask_issue: bool = False
) -> None:
    """
    Save the processed cloud mask to a new file.

    Args:
        ref_mask_path (str): The file path to the reference cloud mask.
        output_path (str): The file path to save the processed cloud mask.
        cloud_mask_issue (bool, optional): Indicates if there is an issue with the cloud mask. Defaults to False.

    Returns:
        None
    """
    # create a boolean cloud mask from the cloud mask
    cloud_mask = process_landsat_cloud_mask(ref_mask_path, cloud_mask_issue)
    with rasterio.open(ref_mask_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(cloud_mask.astype(rasterio.uint8), 1)
    return output_path


def apply_cloudmask_to_dir(input_dir,suffix="_3B_udm2_clip.tif",output_suffix="_combined_mask.tif"):
    """
    Apply cloud and shadow mask to all TIFF files in the given directory.

    This function works specifically for the UDM files from the PlanetScope dataset.
    It creates a combined cloud and shadow mask from the given UDM file.
    Specifically it expects cloud mask at band 6 and the no data mask at band 8.

    Args:
        input_dir (str): The path to the directory containing the TIFF files.
        suffix (str): The suffix of the TIFF files to be processed. The default is "_3B_udm2_clip.tif".


    Returns:
        None
    """
    for file in os.listdir(input_dir):
        if file.endswith(suffix):
            udm_path = os.path.join(input_dir, file)
            print(f"Processing {udm_path}")
            create_cloud_and_nodata_mask(udm_path,output_suffix)
            # create_cloud_and_shadow_mask(udm_path)


def load_udm(udm_filename):
    """Load single-band bit-encoded UDM as a 2D array

    Source:
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb

    """
    with rasterio.open(udm_filename, "r") as src:
        udm = src.read()[0, ...]
    return udm


def create_cloud_and_nodata_mask(udm_path,output_suffix="_combined_mask.tif"):
    """
    Create a combined cloud and shadow mask from the given UDM (User-Defined Mask) file.

    Parameters:
    udm_path (str): The path to the UDM file.
    output_suffix (str): The suffix to be to the name combined mask file.

    Returns:
    numpy.ndarray: The combined cloud and shadow mask.

    Raises:
    FileNotFoundError: If the UDM file does not exist.
    """
    print(f"Creating cloud and shadow mask from {udm_path}")
    with rasterio.open(udm_path) as src:
        cloud_mask = src.read(6)
        cloud_shadow_mask = src.read(3)
        im_nodata = src.read(8)
        # combine the cloud and shadow mask
        combined_mask = np.logical_or(
            np.logical_or(cloud_mask, cloud_shadow_mask), im_nodata
        )

        # write the combined mask to a new file
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

        # write to a new file
        # mask_out = combined_mask > 0

        base_filename = os.path.basename(udm_path.split(".")[-2])
        combined_path = os.path.join(
            os.path.dirname(udm_path), base_filename + output_suffix
        )
        with rasterio.open(combined_path, "w", **profile) as dst:
            print(f"Saving combined mask to {combined_path}")
            dst.write(combined_mask.astype(rasterio.uint8), 1)

        return combined_mask


# def create_cloud_and_shadow_mask(udm_path):
#     """
#     Create a combined cloud and shadow mask from the given UDM (User-Defined Mask) file.

#     Parameters:
#     udm_path (str): The path to the UDM file.

#     Returns:
#     numpy.ndarray: The combined cloud and shadow mask.

#     Raises:
#     FileNotFoundError: If the UDM file does not exist.
#     """
#     print(f"Creating cloud and shadow mask from {udm_path}")
#     with rasterio.open(udm_path) as src:
#         cloud_mask = src.read(6)
#         cloud_shadow_mask = src.read(3)

#         # combine the cloud and shadow mask
#         combined_mask = np.logical_or(cloud_mask, cloud_shadow_mask)

#         # write the combined mask to a new file
#         profile = src.profile
#         profile.update(dtype=rasterio.uint8, count=1)

#         # write to a new file
#         # mask_out = combined_mask > 0

#         base_filename = os.path.basename(udm_path.split('.')[-2])
#         combined_path = os.path.join(os.path.dirname(udm_path), base_filename+'_combined_mask.tif')
#         # combined_bool_path = os.path.join(os.path.dirname(udm_path), base_filename+'_combined_mask_bool.tif')

#         # with rasterio.open(combined_bool_path, 'w', **profile) as dst:
#         #     print(f"Saving combined mask to {combined_bool_path}")
#         #     dst.write(mask_out.astype(rasterio.uint8), 1)

#         with rasterio.open(combined_path, 'w', **profile) as dst:
#             print(f"Saving combined mask to {combined_path}")
#             dst.write(combined_mask.astype(rasterio.uint8), 1)

#         return combined_mask
