
import os
import rasterio
import numpy as np

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

def convert_landsat_to_model_format(input_file: str, output_file: str) -> None:
    """Process the raster file by normalizing and reordering its bands, and save the output.
    It removes the 5th band (SWIR) and reorders the bands to RGBN (red, green, blue, NIR).

    This is used for 4 band planet imagery that needs to be reordered to RGBN from BGRN for the zoo model.

    Args:
        input_file (str): The file path to the input raster file.
        output_file (str): The file path to save the processed raster file.

    Reads the input raster file, normalizes its bands to the range [0, 255],
    reorders the bands to RGB followed by the remaining bands, and saves the
    processed raster to the specified output file.

    The function assumes the input raster has at least three bands (blue, green, red)
    and possibly additional bands.

    Prints the min and max values of the original and normalized red, green, and blue bands,
    and the shape of the reordered bands array.
    """
    # remove the output file if exists just in case it was corrupted (opening with rasterio will fail)
    if os.path.exists(output_file):
        os.remove(output_file)

    with rasterio.open(input_file) as src:
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
        # reordered_bands = np.dstack([band1_normalized, band2_normalized,band3_normalized,] + other_bands_normalized)
        reordered_bands = np.dstack(
            [
                band3_normalized,
                band2_normalized,
                band1_normalized,
            ]
            + other_bands_normalized
        )
        reordered_bands = reordered_bands[:, :, :3]
        # Get the metadata
        meta = src.meta.copy()

        # print(
        #     f"dtype: {reordered_bands.dtype}, shape: {reordered_bands.shape}, count: {reordered_bands.shape[2]}"
        # )
        # Update the metadata to reflect the number of layers and data type
        meta.update(
            {
                "count": reordered_bands.shape[2],
                "dtype": reordered_bands.dtype,
                "driver": "GTiff",
            }
        )
        # make a numpy array that can be saved as a jpg
        # convert_tiff_to_jpg(reordered_bands, input_file)

        # Save the image
        with rasterio.open(output_file, "w", **meta) as dst:
            for i in range(reordered_bands.shape[2]):
                dst.write(reordered_bands[:, :, i], i + 1)

    return output_file

def create_tiled_raster(src_filepath, dst_filepath, block_size=256):
    """
    Create a new tiled raster file with the specified block size.

    Parameters:
        src_filepath (str): The file path of the source raster file.
        dst_filepath (str): The file path of the destination raster file.
        block_size (int, optional): The size of the blocks for tiling. Defaults to 256.

    Returns:
        dst_filepath (str): The file path of the destination raster file.

    Raises:
        None
    """


    # Open the source file
    with rasterio.open(src_filepath) as src:
        # Copy the profile from the source
        profile = src.profile

        # Update the profile to set tiling and block sizes
        profile.update(
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            compress="lzw",  # Optional: Add compression to reduce file size
        )

        # remove the destination file if it already exists just in case it was corrupted
        if os.path.exists(dst_filepath):
            os.remove(dst_filepath)

        # Create a new file with the updated profile
        with rasterio.open(dst_filepath, "w", **profile) as dst:
            # Copy data from the source file to the destination file
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                dst.write(band_data, i)
    return dst_filepath

def convert_from_float_to_unit16(input_path, output_path):
    """
    Process a raster file by performing the following steps:
    1. Open the input raster file.
    2. Modify the data by replacing infinite values with 0 and scaling the data by a factor of 10000.
    3. Create a new output raster file with the modified data.

    Args:
        input_path (str): The path to the input raster file.
        output_path (str): The path to save the output raster file.

    Returns:
        str: The path to the output raster file.
    """
    if os.path.isfile(output_path) and os.path.exists(output_path):
        os.remove(output_path)

    with rasterio.open(input_path) as input_raster:
        out_meta = input_raster.meta.copy()
        modified_data = np.empty(
            (input_raster.count, input_raster.height, input_raster.width),
            dtype=out_meta["dtype"],
        )
        scale = 10000
        for i in range(1, input_raster.count + 1):
            data = input_raster.read(i)
            data[np.isinf(data)] = 0
            data = data * scale
            modified_data[i - 1] = data

        out_meta.update(dtype=rasterio.uint16, count=input_raster.count, nodata=0)

    with rasterio.open(output_path, "w", **out_meta) as output_raster:
        output_raster.write(modified_data)
    return output_path

def format_landsat_tiff(landsat_path: str, output_dir:str="") -> str:
    """
    Formats a Landsat TIFF file by converting it from float to uint16, changing the blocksize to 256x256,
    creating a tiled raster, and converting it to a 3 band RGB ordered TIFF.

    Args:
        landsat_path (str): The file path of the Landsat TIFF file.
        output_dir (str): The directory to save the formatted Landsat TIFF file.

    Returns:
        str: The file path of the formatted Landsat TIFF file.
    """
    # Output file path
    tmp_path = landsat_path.replace("_ms.tif", "_temp.tif")
    landsat_processed_path = landsat_path.replace("_ms.tif", "_processed.tif")
    # convert the landsat from float to uint16 and change the blocksize to 256x256
    tmp_path = convert_from_float_to_unit16(landsat_path, tmp_path)
    # create a tiled raster of the landsat image
    landsat_processed_path = create_tiled_raster(tmp_path, landsat_processed_path)
    output_path = landsat_processed_path.replace(".tif", "_model_format.tif")
    # create the output path at the output directory if specified
    if output_dir:
         output_path = os.path.join(output_dir, os.path.basename(output_path))
    # convert the landsat to a 3 band RGB ordered tiff
    convert_landsat_to_model_format(landsat_processed_path, output_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(landsat_processed_path):
        os.remove(landsat_processed_path)
    return output_path

# Define the input directory containing the Landsat TIFF files
# input_dir = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime07-18-24__07_43_56\L8\ms"
input_dir = r"C:\Users\sf230\Downloads\AkSpit\raw_landsat"
# Define the output directory to save the processed files
output_dir = r"C:\Users\sf230\Downloads\AkSpit\landsat_model_format"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each TIFF file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        landsat_path = os.path.join(input_dir, filename)
        # Process the Landsat TIFF file
        landsat_processed_path = format_landsat_tiff(landsat_path, output_dir)
        print(f"Processed and saved {landsat_path} to {landsat_processed_path}")


