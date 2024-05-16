import rasterio
import numpy as np


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
            compress='lzw'  # Optional: Add compression to reduce file size
        )
        
        # Create a new file with the updated profile
        with rasterio.open(dst_filepath, 'w', **profile) as dst:
            # Copy data from the source file to the destination file
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                dst.write(band_data, i)
                
    print(f'Created a new tiled raster file with block size {block_size}x{block_size} at {dst_filepath}')
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
    with rasterio.open(input_path) as input_raster:
        out_meta = input_raster.meta.copy()
        modified_data = np.empty((input_raster.count, input_raster.height, input_raster.width), dtype=out_meta['dtype'])
        scale = 10000
        for i in range(1, input_raster.count + 1):
            data = input_raster.read(i)
            data[np.isinf(data)] = 0
            data = data * scale
            modified_data[i-1] = data

        out_meta.update(
            dtype=rasterio.uint16,
            count=input_raster.count,
            nodata=0
        )

    print(f"output_path: {output_path}")
    with rasterio.open(output_path, 'w', **out_meta) as output_raster:
        output_raster.write(modified_data)
    return output_path

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

def convert_planet_to_model_format(input_file: str, output_file: str) -> None:
    """Process the raster file by normalizing and reordering its bands, and save the output.
    
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
    with rasterio.open(input_file) as src:
        # Read the bands
        band1 = src.read(1)  # blue
        band2 = src.read(2)  # green
        band3 = src.read(3)  # red
        other_bands = [src.read(i) for i in range(4, src.count + 1)]

        print(f"red min: {np.min(band3)}, red max: {np.max(band3)}")
        print(f"green min: {np.min(band2)}, green max: {np.max(band2)}")
        print(f"blue min: {np.min(band1)}, blue max: {np.max(band1)}")

        # Normalize the bands
        band1_normalized = normalize_band(band1)
        band2_normalized = normalize_band(band2)
        band3_normalized = normalize_band(band3)
        other_bands_normalized = [normalize_band(band) for band in other_bands]

        print(f"red min: {np.min(band3_normalized)}, red max: {np.max(band3_normalized)}")
        print(f"green min: {np.min(band2_normalized)}, green max: {np.max(band2_normalized)}")
        print(f"blue min: {np.min(band1_normalized)}, blue max: {np.max(band1_normalized)}")

        # Reorder the bands RGB and other bands
        reordered_bands = np.dstack([band3_normalized, band2_normalized, band1_normalized] + other_bands_normalized)

        # Get the metadata
        meta = src.meta.copy()

        # Update the metadata to reflect the number of layers and data type
        meta.update({
            "count": reordered_bands.shape[2],
            "dtype": reordered_bands.dtype,
            "driver": 'GTiff'
        })

        # Save the image
        with rasterio.open(output_file, 'w', **meta) as dst:
            for i in range(reordered_bands.shape[2]):
                dst.write(reordered_bands[:, :, i], i + 1)
                

    return output_file
