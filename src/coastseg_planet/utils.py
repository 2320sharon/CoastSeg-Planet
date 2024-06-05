import os
import numpy as np
import glob
import pandas as pd
import shutil
from scipy.ndimage import zoom


def move_files(source_dir: str, output_dir: str, file_identifier: str, move: bool = True) -> None:
    """
    Move or copy files from the source directory to the output directory based on the file identifier.

    Args:
        source_dir (str): The directory where the files are located.
        output_dir (str): The directory where the files will be moved or copied to.
        file_identifier (str): The identifier used to match the files to be moved or copied.
        move (bool, optional): If True, the files will be moved. If False, the files will be copied. 
            Defaults to True.

    Returns:
        None
    """
    for ext in ["*_metadata.json", "*_metadata_clip.xml", "*_udm*.tif"]:
        files = glob.glob(os.path.join(source_dir, f"{file_identifier}{ext}"))
        if files:
            file_name = os.path.basename(files[0])
            output_path = os.path.join(output_dir, file_name)
            if move:
                shutil.move(files[0], output_path)
            else:
                shutil.copyfile(files[0], output_path)

def sort_images(inference_df_path:str,
                output_folder:str,
                move:bool=True,
                move_additional_files:bool=True)->None:
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    move (bool): if True, move the images to the good and bad folders, if False, copy the images
    """
    bad_dir = os.path.join(output_folder, 'bad')
    good_dir = os.path.join(output_folder, 'good')
    dirs = [output_folder, bad_dir, good_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path) 
        file_identifier = '_'.join(im_name.split("_")[:4])
        if inference_df['im_classes'].iloc[i] == 'good':
            output_image_path = os.path.join(good_dir, im_name)
        else:
            output_image_path = os.path.join(bad_dir, im_name)
        if move_additional_files:
            move_files(os.path.dirname(input_image_path), os.path.dirname(output_image_path), file_identifier, move)
        if move:
            shutil.move(input_image_path, output_image_path)
        else:
            shutil.copyfile(input_image_path, output_image_path)

def resize_array(array:np.ndarray, new_size:tuple) -> np.ndarray:
    """
    Resize the input array to the specified new size using cubic interpolation.

    Parameters:
    array (ndarray): The input array to be resized.
    new_size (tuple): The desired new size of the array.

    Returns:
    ndarray: The resized array.

    """
    # calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(new_size, array.shape)]
    # apply the zoom to resize the array
    resized_array = zoom(array, zoom_factors, order=3)  # order=3 corresponds to cubic interpolation
    return resized_array