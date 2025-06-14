import os
import re
import glob
import json
import shutil
import datetime
import configparser
from scipy.ndimage import zoom
import geopandas as gpd
import pandas as pd
from planet import Auth
import numpy as np
from coastseg_planet.processing import get_tiffs_with_bad_area
from json import JSONEncoder


def read_config(config_path: str = None, api_key_override: str = None):
    """
    Main function to read config and set up the API key.

    Args:
        config_path (str): Optional path to a config.ini file.
        api_key_override (str): Optional API key provided directly.

    Returns:
        configparser.ConfigParser: Parsed config object.
    """
    config = _read_config_from_file(config_path)

    if api_key_override:
        _set_api_key(api_key_override)
    else:
        _set_api_key_from_config(config)

    return config


def _read_config_from_file(config_path: str = None):
    """Reads configuration from the given path or a default fallback."""
    config_filepath = config_path or os.path.join(os.getcwd(), "config.ini")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config file not found at {config_filepath}")

    config = configparser.ConfigParser()
    config.read(config_filepath)
    return config


def _set_api_key_from_config(config):
    """Sets API_KEY from config to environment and validates it."""
    api_key = config.get("DEFAULT", "API_KEY", fallback="")
    if not api_key:
        raise ValueError(
            "API_KEY not found in config file. "
            "Please enter your API key in the config file or pass it directly."
        )
    _set_api_key(api_key)


def _set_api_key(api_key: str):
    """Sets the API key in environment and initializes auth."""
    os.environ["API_KEY"] = api_key
    auth = Auth.from_env("API_KEY")
    auth.store()


def find_file_by_regex(
    search_path: str, search_pattern: str = r"^config\.json$"
) -> str:
    """Searches for a file with matching regex in the specified directory

    Args:
        search_path (str): the directory path to search for the  file matching the search pattern
        search_pattern (str): the regular expression pattern to search for the config file

    Returns:
        str: the file path that matched the search_pattern

    Raises:
        FileNotFoundError: if a file is not found in the specified directory
    """
    config_regex = re.compile(search_pattern, re.IGNORECASE)
    for file in os.listdir(search_path):
        if config_regex.match(file):
            file_path = os.path.join(search_path, file)
            return file_path

    raise FileNotFoundError(
        f"file matching pattern {search_pattern} was not found at {search_path}"
    )


def load_data_from_json(filepath: str) -> dict:
    """
    Reads data from a JSON file and returns it as a dictionary.

    The function reads the data from the specified JSON file using the provided filepath.
    It applies a custom object hook, `DecodeDateTime`, to decode the datetime and shoreline
    data if they exist in the dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Data read from the JSON file as a dictionary.

    """

    def DecodeDateTime(readDict):
        """
        Helper function to decode datetime and shoreline data in the dictionary.

        Args:
            readDict (dict): Dictionary to decode.

        Returns:
            dict: Decoded dictionary.

        """
        if "dates" in readDict:
            tmp = [
                datetime.datetime.fromisoformat(dates) for dates in readDict["dates"]
            ]
            readDict["dates"] = tmp
        if "shorelines" in readDict:
            tmp = [
                np.array(shoreline) if len(shoreline) > 0 else np.empty((0, 2))
                for shoreline in readDict["shorelines"]
            ]
            readDict["shorelines"] = tmp
        return readDict

    with open(filepath, "r") as fp:
        data = json.load(fp, object_hook=DecodeDateTime)
    return data


def save_to_json(data: dict, filepath: str) -> None:
    """
    Serializes a dictionary to a JSON file, handling special serialization for datetime and numpy ndarray objects.

    The function handles two special cases:
    1. If the data contains datetime.date or datetime.datetime objects, they are serialized to their ISO format.
    2. If the data contains numpy ndarray objects, they are converted to lists before serialization.

    Parameters:
    - data (dict): Dictionary containing the data to be serialized to a JSON file.
    - filepath (str): Path (including filename) where the JSON file should be saved.

    Returns:
    - None

    Note:
    - This function requires the json, datetime, and numpy modules to be imported.
    """

    class DateTimeEncoder(JSONEncoder):
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()
            # Check for numpy arrays
            if isinstance(obj, np.ndarray):
                # Check if the dtype is 'object', which indicates it might have mixed types including datetimes
                if obj.dtype == "object":
                    # Convert each element of the array
                    return [self.default(item) for item in obj]
                else:
                    # If it's not an object dtype, simply return the array as a list
                    return obj.tolist()

    with open(filepath, "w") as fp:
        json.dump(data, fp, cls=DateTimeEncoder)


def get_matching_files(
    planet_dir, patterns=set(["*AnalyticMS_clip.tif", "*3B_AnalyticMS_toar_clip.tif"])
):
    """
    Get a list of matching files in the specified directory based on the given patterns.

    Args:
        planet_dir (str): The directory path where the files are located.
        patterns (set, optional): A set of file patterns to match against. Defaults to {"*AnalyticMS_clip.tif", "*3B_AnalyticMS_toar_clip.tif"}.

    Returns:
        list: A list of matching file paths.
    """
    matching_files = []
    for pattern in patterns:
        tif_files = glob.glob(os.path.join(planet_dir, pattern))
        matching_files.extend(tif_files)

    return matching_files


def filter_files_by_area(
    directory: str, threshold=0.90, roi_path="", expected_area_km=None, verbose=False
):
    """
    Filters files in a directory based on their area.

    Args:
        directory (str): The directory path where the files are located.
        threshold (float, optional): The threshold value for determining if a file's area is considered bad. Defaults to 0.90.
        roi_path (str, optional): The path to the ROI GeoJSON file used to calculate the expected area. Defaults to "".
        expected_area_km (float, optional): The expected area in square kilometers. If not provided, it will be calculated from the ROI GeoJSON file. Defaults to None.
        verbose (bool, optional): If True, prints additional information during the filtering process. Defaults to False.

    Returns:
        None

    Raises:
        None

    """
    if expected_area_km is None:
        if roi_path == "":
            print(
                "Please provide a path to the ROI GeoJSON file to filter the files by area."
            )
            return
        _, sq_km_area = calculate_area_in_sq_km(roi_path)
    _, sq_km_area = calculate_area_in_sq_km(roi_path)
    tif_files = get_matching_files(directory)
    if not tif_files:
        print(f"No matching files to sort found in the directory: {directory}")
        return
    bad_tiff_paths = get_tiffs_with_bad_area(
        tif_files, sq_km_area, threshold, use_threads=True
    )
    if verbose:
        print(
            f"Moving {len(bad_tiff_paths)} bad tifs to a new folder called bad in the directory: {directory}"
        )
    # for each bad tiff move it a new folder called bad
    bad_path = os.path.join(directory, "bad")
    # I need to move the corresponding files to the bad folder (xml, udm tif, and json)
    os.makedirs(bad_path, exist_ok=True)
    for bad_tiff in bad_tiff_paths:
        try:
            im_name = os.path.basename(bad_tiff)
            file_identifier = "_".join(im_name.split("_")[:4])
            move_files(directory, bad_path, file_identifier, move=True)
        except Exception as e:
            print(f"Error moving {bad_tiff} to {bad_path}")
            print(e)


def calculate_area_in_sq_km(geojson_path: str) -> tuple:
    """
    Calculates the total area in square kilometers from a GeoJSON file.

    Args:
        geojson_path (str): The path to the GeoJSON file.

    Returns:
        tuple: A tuple containing the total area in square meters and square kilometers.
    """
    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Check the CRS of the GeoDataFrame
    if not gdf.crs.is_projected:
        # Reprojecting to UTM (automatic UTM zone selection based on ROI centroid)
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Calculate the area in square meters
    gdf["area_sqm"] = gdf.geometry.area

    # Calculate the area in square kilometers
    gdf["area_sqkm"] = gdf["area_sqm"] / 1e6

    # Sum the areas to get total area in sqm and sqkm
    total_area_sqm = gdf["area_sqm"].sum()
    total_area_sqkm = gdf["area_sqkm"].sum()

    return total_area_sqm, total_area_sqkm


def get_file_path(
    directory: str, base_filename: str, regex: str = "*udm2_clip_combined_mask.tif"
):
    """
    Get the file path of the cloud mask for the planet image.

    Args:
        directory (str): The directory where the cloud masks are located.
        base_filename (str): The base filename of the planet image.
        regex (str, optional): The regular expression pattern to match the cloud mask filenames. Defaults to "*udm2_clip_combined_mask.tif".

    Returns:
        str or None: The file path of the cloud mask if found, otherwise None.
    """
    cloud_masks_found = glob.glob(os.path.join(directory, f"{base_filename}{regex}"))
    if cloud_masks_found:
        return cloud_masks_found[0]
    return None


def move_files(
    source_dir: str, output_dir: str, file_identifier: str, move: bool = True
) -> None:
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ext in [
        "*_metadata.json",
        "*_metadata_clip.xml",
        "*_udm*.tif",
        "*.tif",
        "*.xml",
    ]:
        pattern = os.path.join(source_dir, f"{file_identifier}{ext}")
        files = glob.glob(pattern)
        for file in files:
            file_name = os.path.basename(file)
            output_path = os.path.join(output_dir, file_name)
            if move:
                shutil.move(file, output_path)
            else:
                shutil.copyfile(file, output_path)


def sort_images(
    inference_df_path: str,
    output_folder: str,
    move: bool = True,
    move_additional_files: bool = True,
) -> None:
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    move (bool): if True, move the images to the good and bad folders, if False, copy the images
    """
    bad_dir = os.path.join(output_folder, "bad")
    good_dir = os.path.join(output_folder, "good")
    dirs = [output_folder, bad_dir, good_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"inference_df_path : {inference_df_path}")
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df["im_paths"].iloc[i]
        im_name = os.path.basename(input_image_path)
        file_identifier = "_".join(im_name.split("_")[:4])
        if inference_df["im_classes"].iloc[i] == "good":
            output_image_path = os.path.join(good_dir, im_name)
        else:
            output_image_path = os.path.join(bad_dir, im_name)

        if move_additional_files:
            move_files(
                os.path.dirname(input_image_path),
                os.path.dirname(output_image_path),
                file_identifier,
                move,
            )

        if not os.path.exists(input_image_path):
            continue

        if move:
            shutil.move(input_image_path, output_image_path)
        else:
            shutil.copyfile(input_image_path, output_image_path)


def resize_array(array: np.ndarray, new_size: tuple) -> np.ndarray:
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
    resized_array = zoom(
        array, zoom_factors, order=3
    )  # order=3 corresponds to cubic interpolation
    return resized_array
