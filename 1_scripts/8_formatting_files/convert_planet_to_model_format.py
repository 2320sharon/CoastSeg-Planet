import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging

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

target_path = r"C:\development\coastseg-planet\downloads\DUCK_NC_cloud_0.8_TOAR_enabled_2022-04-01_to_2024-04-01\834c61f8-df4c-47d0-aafa-b521fc81cc27\PSScene\good\20230919_153544_28_24a4_3B_AnalyticMS_toar_clip.tif"
directory = os.path.dirname(target_path)
print(directory)
separator = "_3B"
output_suffix="_TOAR_model_format.tif"
base_filename = get_base_filename(target_path, separator)
output_path = os.path.join(
    directory, f"{base_filename}{separator}{output_suffix}"
)

convert_planet_to_model_format(target_path, output_path)

with rasterio.open(output_path) as src:
    print(src.meta)
    print(src.count)