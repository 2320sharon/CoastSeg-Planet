
import os
import rasterio
import numpy as np

def apply_cloudmask_to_dir(input_dir):
    """
    Apply cloud and shadow mask to all TIFF files in the given directory.

    Args:
        input_dir (str): The path to the directory containing the TIFF files.

    Returns:
        None
    """
    for file in os.listdir(input_dir):
        if file.endswith("3B_udm2_clip.tif"):
            udm_path = os.path.join(input_dir, file)
            print(f"Processing {udm_path}")
            create_cloud_and_shadow_mask(udm_path)
            
def load_udm(udm_filename):
    '''Load single-band bit-encoded UDM as a 2D array
    
    Source: 
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/udm/udm.ipynb
        
    '''
    with rasterio.open(udm_filename, 'r') as src:
        udm = src.read()[0,...]
    return udm

def create_cloud_and_shadow_mask(udm_path):
    """
    Create a combined cloud and shadow mask from the given UDM (User-Defined Mask) file.

    Parameters:
    udm_path (str): The path to the UDM file.

    Returns:
    numpy.ndarray: The combined cloud and shadow mask.

    Raises:
    FileNotFoundError: If the UDM file does not exist.
    """
    print(f"Creating cloud and shadow mask from {udm_path}")
    with rasterio.open(udm_path) as src:
        cloud_mask = src.read(6)
        cloud_shadow_mask = src.read(3)
        
        # combine the cloud and shadow mask
        combined_mask = np.logical_or(cloud_mask, cloud_shadow_mask)
        
        # write the combined mask to a new file
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        
        # write to a new file
        # mask_out = combined_mask > 0
        
        base_filename = os.path.basename(udm_path.split('.')[-2])
        combined_path = os.path.join(os.path.dirname(udm_path), base_filename+'_combined_mask.tif')
        # combined_bool_path = os.path.join(os.path.dirname(udm_path), base_filename+'_combined_mask_bool.tif')
        
        # with rasterio.open(combined_bool_path, 'w', **profile) as dst:
        #     print(f"Saving combined mask to {combined_bool_path}")
        #     dst.write(mask_out.astype(rasterio.uint8), 1)
        
        with rasterio.open(combined_path, 'w', **profile) as dst:
            print(f"Saving combined mask to {combined_path}")
            dst.write(combined_mask.astype(rasterio.uint8), 1)
        
        return combined_mask