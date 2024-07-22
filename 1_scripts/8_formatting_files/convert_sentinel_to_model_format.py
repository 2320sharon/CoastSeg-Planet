import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
import os
import glob
import tqdm

# the order of the bands download by coastseg/coastsat for each satellite
# for  S2 it is Blue (B2) Green (B3) Red (B4) NIR (B8) SWIR1 (B11) 
# see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED#bands
    # bands_dict = {
    #     "L5": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
    #     "L7": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
    #     "L8": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
    #     "L9": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
    #     "S2": ["B2", "B3", "B4", "B8", "s2cloudless", "B11", "QA60"],
    # }


def convert_directory_to_model_format(
    directory: str,
    input_suffix: str = "*_TOAR.tif",
    output_suffix: str = "_TOAR_model_format.tif",
    separator="_3B",
    crs: int = None,
    verbose: bool = False,
    number_of_bands: int = 3,
    save_path: str = None,
):
    """
    Convert all files with a specific suffix in a directory to a specific TOAR model format.

    Args:
        directory (str): The directory path where the files are located.
        input_suffix (str, optional): The suffix of the input files to be converted. Defaults to '*_TOAR.tif'.
        output_suffix (str, optional): The suffix of the output files in the TOAR model format. Defaults to '_TOAR_model_format.tif'.
        separator (str, optional): The separator used in the filenames. Defaults to '_3B'.
        crs (int, optional): The coordinate reference system to use for the output files. Defaults to None.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
        number_of_bands (int, optional): Number of bands for the output files. Defaults to 3.
        save_path (str, optional): Directory to save the converted files. Defaults to None.

    Returns:
        None
    """
    if not save_path:
        save_path = directory

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    input_paths = glob.glob(os.path.join(directory, input_suffix))
    if not input_paths:
        print(f"No files found with suffix {input_suffix} in {directory}")
        return

    for target_path in tqdm.tqdm(input_paths, desc="Converting files to model format"):
        base_filename = get_base_filename(target_path, separator)
        output_path = os.path.join(save_path, f"{base_filename}{separator}{output_suffix}")

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            convert_planet_to_model_format(target_path, output_path, number_of_bands=number_of_bands, crs=crs)
            if verbose:
                print(f"Converted {target_path} to {output_path}")
        except Exception as e:
            print(f"Error converting {target_path}: {e}")


def get_log_file_name(base_name="conversion.log"):
    """Generate a new log file name with an incremented number if the base name already exists."""
    if not os.path.exists(base_name):
        return base_name
    base, ext = os.path.splitext(base_name)
    counter = 1
    new_name = f"{base}_{counter}{ext}"
    while os.path.exists(new_name):
        counter += 1
        new_name = f"{base}_{counter}{ext}"
    return new_name

def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize the band to the range [0, 255].

    Args:
        band (np.ndarray): The input band to normalize.

    Returns:
        np.ndarray: The normalized band.
    """
    min_val = np.min(band)
    max_val = np.max(band)
    normalized_band = ((band - min_val) / (max_val - min_val)) * 255
    return normalized_band.astype(np.uint8)


def convert_planet_to_model_format(
    input_file: str, output_file: str, number_of_bands: int = 3, crs=None
) -> None:
    """Process the raster file by normalizing and reordering its bands, and save the output.

    This is used for 4 band planet imagery that needs to be reordered to RGBN from BGRN for the zoo model.

    Args:
        input_file (str): The file path to the input raster file.
        output_file (str): The file path to save the processed raster file.
        number_of_bands (int): The number of bands to keep in the output.
        crs (dict or str, optional): The target coordinate reference system.

    Reads the input raster file, normalizes its bands to the range [0, 255],
    reorders the bands to RGB followed by the remaining bands, and saves the
    processed raster to the specified output file.

    The function assumes the input raster has at least three bands (blue, green, red)
    and possibly additional bands.

    Prints the min and max values of the original and normalized red, green, and blue bands,
    and the shape of the reordered bands array.
    """
    temp_file = "reprojected_temp.tif"
    log_file = get_log_file_name("conversion.log")
    logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

    try:
        if crs:
            logging.info(f"Reprojecting the raster to {crs}")
            with rasterio.open(input_file) as src:
                if src.crs == crs:
                    logging.info(f"The raster is already in the target CRS: {crs}")
                    reprojected_file = input_file
                else:
                    # Calculate the transform and dimensions for the new CRS
                    transform, width, height = calculate_default_transform(
                        src.crs, crs, src.width, src.height, *src.bounds
                    )
                    meta = src.meta.copy()
                    meta.update(
                        {"crs": crs, "transform": transform, "width": width, "height": height}
                    )

                    with rasterio.open(temp_file, "w", **meta) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=crs,
                                resampling=Resampling.nearest,
                            )

                    reprojected_file = temp_file
        else:
            reprojected_file = input_file

        # this prevents issues if the output file already exists and is corrupted
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                logging.info(f"Existing output file {output_file} deleted.")
            except Exception as e:
                logging.error(f"Could not delete existing output file {output_file}: {e}")
                raise e

        with rasterio.open(reprojected_file) as src:
            # Read the bands
            band1 = src.read(1)  # blue
            band2 = src.read(2)  # green
            band3 = src.read(3)  # red
            other_bands = [src.read(i) for i in range(4, src.count + 1)]

            # Normalize the bands
            band1_normalized = normalize_band(band1)
            band2_normalized = normalize_band(band2)
            band3_normalized = normalize_band(band3)
            other_bands_normalized = [normalize_band(band) for band in other_bands]

            # Reorder the bands RGB and other bands
            reordered_bands = np.dstack(
                [band3_normalized, band2_normalized, band1_normalized]
                + other_bands_normalized
            )
            reordered_bands = reordered_bands[:, :, :number_of_bands]

            # Get the metadata
            meta = src.meta.copy()

            # Update the metadata to reflect the number of layers and data type
            meta.update(
                {
                    "count": reordered_bands.shape[2],
                    "dtype": reordered_bands.dtype,
                    "driver": "GTiff",
                }
            )



            # Save the image
            with rasterio.open(output_file, "w", **meta) as dst:
                for i in range(reordered_bands.shape[2]):
                    dst.write(reordered_bands[:, :, i], i + 1)

        # Delete the temporary file if it was created
        if crs and os.path.exists(temp_file):
            os.remove(temp_file)

        logging.info(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}", exc_info=True)
        raise e


def get_base_filename(tiff_path, separator="_3B"):
    if separator not in os.path.basename(tiff_path):
        raise ValueError(f"Separator '{separator}' not found in '{tiff_path}'")
    return os.path.basename(tiff_path).split(separator, 1)[0]

# make the output path



# target_path = r"D:\3_development\imreg_dft\imreg_dft\AK_spit\sentinel_experiment\template"
# directory = os.path.dirname(target_path)
input_directory = r"C:\Users\sf230\Downloads\Santa_Cruz\s2\raw_s2"
directory = r"C:\Users\sf230\Downloads\Santa_Cruz\s2\s2_model_format"
separator = "_S2"
output_suffix="_TOAR_model_format.tif"

convert_directory_to_model_format(input_directory,
                                  input_suffix='*_ms.tif',
                                    separator=separator,
                                    number_of_bands=3,
                                    verbose=True,
                                    save_path=directory)

