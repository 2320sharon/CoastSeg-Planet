# Standard library imports
import glob
import os
from datetime import datetime
from typing import List
from line_profiler import profile

# Third-party library imports
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import skimage.measure as measure
import skimage.morphology as morphology
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiLineString, MultiPoint
from skimage import transform
from tqdm import tqdm

from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Local module imports
from coastseg.extracted_shoreline import (
    get_class_mapping,
    get_indices_of_classnames,
    load_merged_image_labels,
    remove_small_objects_and_binarize,
)
from coastseg_planet import model, processing, utils,plotting
from coastseg_planet.plotting import (
    create_legend,
    plot_image_with_legend,
    save_detection_figure,
)
from coastseg_planet.processing import (
    get_epsg_from_tiff,
    get_georef,
    read_planet_tiff,
)
from coastsat import SDS_tools


def get_utm_zone_from_geotiff(file_path):
    """
    Get the UTM zone from a GeoTIFF file.

    Parameters:
    - file_path (str): Path to the GeoTIFF file.

    Returns:
    - utm_zone (str): UTM zone as EPSG code (e.g., 'EPSG:32633') or None if not a UTM CRS.
    """
    with rasterio.open(file_path) as dataset:
        # Get the CRS of the dataset
        crs = dataset.crs
        # Check if the CRS is a projected CRS and is in UTM
        if crs.is_projected :
            # Extract the EPSG code
            epsg_code = crs.to_epsg()
            return f"EPSG:{epsg_code}"
        else:
            print("The CRS is not in UTM format.")
            return None

def dilate_shoreline(gdf, buffer_size, output_file, pixel_size=3, utm_zone=None):
    """
    Dilate a shoreline (or any linestring geometry) by a specified buffer size in meters,
    and save the rasterized dilated geometry as a TIFF file.

    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame containing linestring geometries.
    - buffer_size (float): Buffer size in meters for dilation.
    - output_file (str): Path to save the output TIFF file.
    - pixel_size (float): Size of each pixel in meters for the output raster. Default is 3 meters.
    - utm_zone (str, optional): UTM zone EPSG code (e.g., 'EPSG:32633'). If None, it will be determined based on the geometry.

    Returns:
    - None
    """
    # Function to determine UTM zone if not provided
    def get_utm_crs(geometry):
        lon = geometry.centroid.x
        utm_zone = int(np.floor((lon + 180) / 6) + 1)
        is_northern = geometry.centroid.y >= 0
        utm_crs = f"EPSG:{32600 + utm_zone}" if is_northern else f"EPSG:{32700 + utm_zone}"
        return utm_crs

    # Determine the UTM CRS
    if utm_zone is None:
        utm_crs = get_utm_crs(gdf.geometry.unary_union)
    else:
        utm_crs = utm_zone

    # Convert to UTM CRS
    gdf_utm = gdf.to_crs(utm_crs)

    # Apply the buffer directly in meters (unit of UTM CRS)
    gdf_utm['geometry'] = gdf_utm.geometry.buffer(buffer_size)

    # Get the bounds of the buffered geometry
    minx, miny, maxx, maxy = gdf_utm.total_bounds

    # Define the raster size based on the pixel size and geometry bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Create an empty raster array
    raster = np.zeros((height, width), dtype=np.uint8)

    # Rasterize the buffered geometries
    shapes = ((geom, 1) for geom in gdf_utm.geometry if geom.is_valid and not geom.is_empty)
    rasterized = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, default_value=1)

    # Prepare metadata for saving
    metadata = {
        'height': rasterized.shape[0],
        'width': rasterized.shape[1],
        'count': 1,
        'dtype': rasterized.dtype,
        'crs': gdf_utm.crs,
        'transform': transform
    }

    # Save the raster to a TIFF file
    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(rasterized, 1)

    print(f"Saved dilated shoreline raster to {output_file}")
    return output_file


def stringify_datetime_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Check if any of the columns in a GeoDataFrame have the type pandas timestamp and convert them to string.

    Args:
        gdf: A GeoDataFrame.

    Returns:
        A new GeoDataFrame with the same data as the original, but with any timestamp columns converted to string.
    """
    timestamp_cols = [
        col for col in gdf.columns if pd.api.types.is_datetime64_any_dtype(gdf[col])
    ]

    if not timestamp_cols:
        return gdf

    gdf = gdf.copy()

    for col in timestamp_cols:
        gdf[col] = gdf[col].astype(str)

    return gdf

def get_multiline_list(geometry):
    """Get list of coordinates from a MultiLineString geometry."""
    lines_list = []
    for line in geometry.geoms:
        lines_list.extend(line.coords)
    return lines_list

def get_multipoint_list(geometry):
    """Get list of coordinates from a MultiPoint geometry."""
    points_list = []
    for point in geometry.geoms:
        points_list.append(list(point.coords[0]))
    return points_list

def convert_shoreline_gdf_to_dict(shoreline_gdf, date_format="%d-%m-%Y", output_crs=None,):
    """
    Convert a GeoDataFrame containing shorelines into a dictionary with dates and shorelines.

    Parameters:
    shoreline_gdf (GeoDataFrame): The input GeoDataFrame with shoreline data.
    date_format (str): The format string for converting dates to strings. Default is "%d-%m-%Y".
    output_crs (str or dict, optional): The target CRS to convert the coordinates to. If None, no conversion is performed.

    Returns:
    dict: A dictionary with keys 'dates' and 'shorelines', where 'dates' is a list of date strings and 'shorelines' is a list of numpy arrays of coordinates.
    """
    shorelines = []
    dates = []

    if output_crs is not None:
        shoreline_gdf = shoreline_gdf.to_crs(output_crs)

    for date, group in shoreline_gdf.groupby('date'):
        date_str = date.strftime(date_format)
        dates.append(date_str)

        shorelines_array = []
        points_list = []
        # for each geometry in the group, get the coordinates as a list of points
        for geometry in group.geometry:
            # add all the points from the geometry to the points_list
            # print(f"geometry type: {geometry.geom_type}")
            if isinstance(geometry ,MultiPoint):
                pts = get_multipoint_list(geometry)
                points_list.extend(pts)
            elif isinstance(geometry, MultiLineString):
                lines = get_multiline_list(geometry)
                points_list.extend(lines)
            else:
                points_list.extend(geometry.coords)

        # put all the points into a single numpy array to represent the shoreline for that date then append to shorelines
        shorelines_array = np.array(points_list)
        shorelines.append(shorelines_array)

    shorelines_dict = {'dates': dates, 'shorelines': shorelines}
    return shorelines_dict

# def convert_shoreline_gdf_to_dict(shoreline_gdf, date_format="%d-%m-%Y", output_crs=None):
#     """
#     Convert a GeoDataFrame containing shorelines into a dictionary with dates and shorelines.

#     Parameters:
#     shoreline_gdf (GeoDataFrame): The input GeoDataFrame with shoreline data.
#     date_format (str): The format string for converting dates to strings. Default is "%d-%m-%Y".
#     output_crs (str or dict, optional): The target CRS to convert the coordinates to. If None, no conversion is performed.

#     Returns:
#     dict: A dictionary with keys 'dates' and 'shorelines', where 'dates' is a list of date strings and 'shorelines' is a list of numpy arrays of coordinates.
#     """
#     shorelines = []
#     dates = []

#     if output_crs is not None:
#         shoreline_gdf = shoreline_gdf.to_crs(output_crs)

#     for idx, row in shoreline_gdf.iterrows():
#         date_str = row.date.strftime(date_format)
#         geometry = row.geometry
#         if geometry is not None:
#             if isinstance(geometry, MultiLineString):
#                 for line in geometry.geoms:
#                     shorelines_array = np.array(line.coords)
#                     shorelines.append(shorelines_array)
#                     dates.append(date_str)
#             else:
#                 shorelines_array = np.array(geometry.coords)
#                 shorelines.append(shorelines_array)
#                 dates.append(date_str)

#     shorelines_dict = {'dates': dates, 'shorelines': shorelines}
#     return shorelines_dict

def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, ref_sl,settings):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is
    given by settings['max_dist_ref'].

    KV WRL 2018

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    ref_sl : np.array
        lat,lon,mean sea level coordinates of the reference shoreline
        ex. np.array([[lat,lon,0],[lat,lon,0]...])
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres

    Returns:
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)
    # convert reference shoreline to pixel coordinates
    ref_sl_conv = SDS_tools.convert_epsg(
        ref_sl, settings["output_epsg"], image_epsg
    )[:, :-1]
    ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
    ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)
    # make sure that the pixel coordinates of the reference shoreline are inside the image
    idx_row = np.logical_and(
        ref_sl_pix_rounded[:, 0] > 0, ref_sl_pix_rounded[:, 0] < im_shape[1]
    )
    idx_col = np.logical_and(
        ref_sl_pix_rounded[:, 1] > 0, ref_sl_pix_rounded[:, 1] < im_shape[0]
    )
    idx_inside = np.logical_and(idx_row, idx_col)
    ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside, :]
    # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
    im_binary = np.zeros(im_shape)
    for j in range(len(ref_sl_pix_rounded)):
        im_binary[ref_sl_pix_rounded[j, 1], ref_sl_pix_rounded[j, 0]] = 1
    im_binary = im_binary.astype(bool)
    # dilate the binary image to create a buffer around the reference shoreline
    max_dist_ref_pixels = np.ceil(settings["max_dist_ref"] / pixel_size)
    se = morphology.disk(max_dist_ref_pixels)
    im_buffer = morphology.binary_dilation(im_binary, se)

    return im_buffer

def make_coastsat_compatible(feature: gpd.geodataframe) -> list:
    """Return the feature as an np.array in the form:
        [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    Args:
        feature (gpd.geodataframe): clipped portion of shoreline within a roi
    Returns:
        list: shorelines in form:
            [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...])
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(index_parts=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: tuple(np.array(row.geometry.coords).tolist()), axis=1
    )
    features = list(lat_lng)
    return features

def get_reference_shoreline_as_array(
    shoreline_gdf: gpd.geodataframe, output_crs: str
) -> np.ndarray:
    """
    Converts a GeoDataFrame of shoreline features into a numpy array of latitudes, longitudes, and zeroes representing the mean sea level.
    Ex. [[lat, lon, 0], [lat, lon, 0], ...]
    Args:
    - shoreline_gdf (GeoDataFrame): A GeoDataFrame of shoreline features.
    - output_crs (str): The output CRS to which the shoreline features need to be projected.

    Returns:
    - np.ndarray: A numpy array of latitudes, longitudes, and zeroes representing the mean sea level.
    """
    # project shorelines's espg from map's espg to output espg given in settings
    reprojected_shorlines = shoreline_gdf.to_crs(output_crs)
    # convert shoreline_in_roi gdf to coastsat compatible format np.array([[lat,lon,0],[lat,lon,0]...])
    shorelines = make_coastsat_compatible(reprojected_shorlines)
    # shorelines = [([lat,lon],[lat,lon],[lat,lon]),([lat,lon],[lat,lon],[lat,lon])...]
    # Stack all the tuples into a single list of n rows X 2 columns
    shorelines = np.vstack(shorelines)
    # Add third column of 0s to represent mean sea level
    shorelines = np.insert(shorelines, 2, np.zeros(len(shorelines)), axis=1)
    
    return shorelines

def load_image_labels(npz_file: str) -> np.ndarray:
    """
    Load in image labels from a .npz file. Loads in the "grey_label" array from the .npz file and returns it as a 2D

    Parameters:
    npz_file (str): The path to the .npz file containing the image labels.

    Returns:
    np.ndarray: A 2D numpy array containing the image labels from the .npz file.
    """
    if not os.path.isfile(npz_file) or not npz_file.endswith(".npz"):
        raise ValueError(f"{npz_file} is not a valid .npz file.")

    data = np.load(npz_file)
    return data["grey_label"]

def extract_contours(filtered_contours_long):
    """
    Extracts x and y coordinates from a list of contours and combines them into a single array.

    Args:
        filtered_contours_long (list): List of contours, where each contour is a numpy array with at least 2 columns.

    Returns:
        np.ndarray: A transposed array with x coordinates in the first column and y coordinates in the second column.
    """
    only_points = [contour[:, :2] for contour in filtered_contours_long]
    x_points = np.array([])
    y_points = np.array([])

    for points in only_points:
        x_points = np.append(x_points, points[:, 0])
        y_points = np.append(y_points, points[:, 1])

    contours_array = np.transpose(np.array([x_points, y_points]))
    return contours_array

def convert_world2pix(
    points: list[np.ndarray] | np.ndarray, georef: np.ndarray
) -> list[np.ndarray] | np.ndarray:
    """
    Converts world coordinates to pixel coordinates.

    Args:
        points (list[np.ndarray] | np.ndarray): List of points or array of points in world coordinates.
        georef (np.ndarray): Georeference information.

    Returns:
        list[np.ndarray] | np.ndarray: Converted points in pixel coordinates.
    """
    aff_mat = np.array(
        [
            [georef[1], georef[2], georef[0]],
            [georef[4], georef[5], georef[3]],
            [0, 0, 1],
        ]
    )
    tform = transform.AffineTransform(aff_mat)
    if isinstance(points, list):
        points_converted = [
            tform.inverse(arr[:, :2]) if arr.ndim == 2 else tform.inverse(arr)
            for arr in points
        ]
    elif isinstance(points, np.ndarray):
        points_converted = tform.inverse(points)
    else:
        raise ValueError("Invalid input type")
    return points_converted


def get_class_indices_from_model_card(npz_file,model_card_path):
    """
        Retrieves the class indices, land mask, and class mapping from the given model card.

        Parameters:
            npz_file (str): The path to the NPZ file.
            model_card_path (str): The path to the model card.
        Returns:
            tuple: A tuple containing the following:
                - water_classes_indices (list): The indices of the water classes.
                - land_mask (ndarray): The land mask.
                - class_mapping (dict): The mapping of class indices to class names.
    """
    # get the water index from the model card
    water_classes_indices = get_indices_of_classnames(
        model_card_path, ["water", "whitewater"]
    )
    # Get the mapping of each label value to its class for example: Sample class mapping {0:'water',  1:'whitewater', 2:'sand', 3:'rock'}
    class_mapping = get_class_mapping(model_card_path)

    # Get the land mask from the npz files. Use the water classes indices to get the water mask
    land_mask = load_merged_image_labels(npz_file, class_indices=water_classes_indices)
    
    return water_classes_indices,land_mask,class_mapping 

def extract_shorelines_with_reference_shoreline(directory:str,
                    suffix:str,
                    reference_shoreline:np.ndarray,
                    extract_shorelines_settings:dict,
                    model_name:str='segformer_RGB_4class_8190958',
                    cloud_mask_suffix:str = '3B_udm2_clip.tif',
                    separator = '_3B'):
    """
    Extracts shorelines from a directory of tiff files using a model and a reference shoreline.

    Args:
        directory (str): The directory containing the tiff files.
        suffix (str): The suffix of the tiff files to extract shorelines from.

        model_name (str): The name of the model to use for image segmentation. Defaults to 'segformer_RGB_4class_8190958'.
        reference_shoreline (np.ndarray): The reference shoreline as a numpy array.
        extract_shorelines_settings (dict): The settings for the shoreline extraction.
        cloud_mask_suffix (str, optional): The suffix of the cloud mask files. Defaults to '3B_udm2_clip.tif'.
        separator (str, optional): The separator used in the tiff file names. Defaults to '_3B'.

    Returns:
        dict: A dictionary containing the extracted shorelines.
            Contains the keys 'dates' and 'shorelines'. 
            'dates' is a list of date strings and 'shorelines' is a list of numpy arrays of coordinates. Each shoreline is associated with its corresponding date.
    """
                       
    # if the directory contains no files with the suffix then return an empty dictionary
    filtered_tiffs = glob.glob(os.path.join(directory, f"*{suffix}.tif"))
    if len(filtered_tiffs) == 0:
        print(f"No tiffs found in the directory: {directory}") 
        return {}                 
                       
    all_extracted_shorelines = []
    counter = 0

    for target_path in tqdm(filtered_tiffs, desc="Extracting Shorelines"):
        base_filename = processing.get_base_filename(target_path, separator)
        # get the processed coregistered file, the cloud mask and the npz file
        cloud_mask_path = utils.get_file_path(directory, base_filename, f"*{cloud_mask_suffix}")
        if not cloud_mask_path:
            print(f"Skipping {os.path.basename(target_path)} because cloud mask with {cloud_mask_suffix} not found")
            continue
        # get the date from the path
        date=processing.get_date_from_path(base_filename)

        # apply the model to the image
        model.apply_model_to_image(target_path,directory)
        # this is the file generated by the model
        npz_path = os.path.join(directory, 'out', f"{base_filename}{suffix}_res.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping {npz_path} because it does not exist")
            continue
            
        save_path = os.path.join(directory, 'shoreline_detection_figures')

        # load the model card containing the model information like label to class mapping
        model_directory  = model.get_model_location(model_name)
        model_card_path = utils.find_file_by_regex(
            model_directory, r".*modelcard\.json$"
        )
                
        # this shoreline is in the UTM projection
        shoreline_gdf = get_shorelines_from_model_reference_shoreline(cloud_mask_path,
                                                  target_path,
                                                  model_card_path,
                                                  npz_path,date,
                                                  satname= 'planet',
                                                settings = extract_shorelines_settings,
                                                reference_shoreline = reference_shoreline,
                                            save_path=save_path)
        
        all_extracted_shorelines.append(shoreline_gdf)
        counter += 1

        # Save the collected shorelines so far
        if counter % 3 == 0 and len(all_extracted_shorelines) > 0:
            shorelines_gdf = concat_and_sort_geodataframes(all_extracted_shorelines, "date")
            extracted_shorelines_path = os.path.join(directory, f"raw_extracted_shorelines.geojson")
            shorelines_gdf.to_file(extracted_shorelines_path, driver="GeoJSON")
            print(f"Saved extracted shorelines to {extracted_shorelines_path}")

    # put all the shorelines into a single geodataframe
    if len(all_extracted_shorelines) == 0:
        print(f"No shorelines extracted from {directory}. Double check the settings and the files in the directory.")
        return {'shorelines':[],'dates':[]}

    shorelines_gdf = concat_and_sort_geodataframes(all_extracted_shorelines, "date")

    # save the geodataframe to a geojson file
    extracted_shorelines_path = os.path.join(directory, f"raw_extracted_shorelines.geojson")  
    stringified_shoreline_gdf = stringify_datetime_columns(shorelines_gdf.copy())
    stringified_shoreline_gdf.to_file(extracted_shorelines_path, driver="GeoJSON")
    print(f"saved extracted shorelines to {extracted_shorelines_path}")

    # convert the geodataframe to a dictionary
    shorelines_dict = convert_shoreline_gdf_to_dict(shorelines_gdf,date_format='%Y-%m-%dT%H:%M:%S%z')
    return shorelines_dict


def extract_shorelines_with_reference_shoreline_gdf(directory:str,
                    suffix:str,
                    reference_shoreline:gpd.geodataframe,
                    extract_shorelines_settings:dict,
                    model_name:str='segformer_RGB_4class_8190958',
                    cloud_mask_suffix:str = '3B_udm2_clip.tif',
                    separator = '_3B'):
    """
    Extracts shorelines from a directory of tiff files using a model and a reference shoreline.

    Args:
        directory (str): The directory containing the tiff files.
        suffix (str): The suffix of the tiff files to extract shorelines from.

        model_name (str): The name of the model to use for image segmentation. Defaults to 'segformer_RGB_4class_8190958'.
        reference_shoreline (gpd.geodataframe): The reference shoreline as a geodataframe.
        extract_shorelines_settings (dict): The settings for the shoreline extraction.
        cloud_mask_suffix (str, optional): The suffix of the cloud mask files. Defaults to '3B_udm2_clip.tif'.
        separator (str, optional): The separator used in the tiff file names. Defaults to '_3B'.

    Returns:
        dict: A dictionary containing the extracted shorelines.
            Contains the keys 'dates' and 'shorelines'. 
            'dates' is a list of date strings and 'shorelines' is a list of numpy arrays of coordinates. Each shoreline is associated with its corresponding date.
    """
                       
    # if the directory contains no files with the suffix then return an empty dictionary
    filtered_tiffs = glob.glob(os.path.join(directory, f"*{suffix}.tif"))
    if len(filtered_tiffs) == 0:
        print(f"No tiffs found in the directory: {directory}") 
        return {}                 
                       
    all_extracted_shorelines = []
    counter = 0

    for target_path in tqdm(filtered_tiffs, desc="Extracting Shorelines"):
        base_filename = processing.get_base_filename(target_path, separator)
        # get the processed coregistered file, the cloud mask and the npz file
        cloud_mask_path = utils.get_file_path(directory, base_filename, f"*{cloud_mask_suffix}")
        if not cloud_mask_path:
            print(f"Skipping {os.path.basename(target_path)} because cloud mask with {cloud_mask_suffix} not found")
            continue
        # get the date from the path
        date=processing.get_date_from_path(base_filename)

        # apply the model to the image
        model.apply_model_to_image(target_path,directory)
        # this is the file generated by the model
        npz_path = os.path.join(directory, 'out', f"{base_filename}{suffix}_res.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping {npz_path} because it does not exist")
            continue
            
        save_path = os.path.join(directory, 'shoreline_detection_figures')

        # load the model card containing the model information like label to class mapping
        model_directory  = model.get_model_location(model_name)
        model_card_path = utils.find_file_by_regex(
            model_directory, r".*modelcard\.json$"
        )
                
        # this shoreline is in the UTM projection
        shoreline_gdf = get_shorelines_from_model_reference_shoreline_gdf(cloud_mask_path,
                                                  target_path,
                                                  model_card_path,
                                                  npz_path,date,
                                                  satname= 'planet',
                                                settings = extract_shorelines_settings,
                                                reference_shoreline_gdf = reference_shoreline,
                                            save_path=save_path)
        
        all_extracted_shorelines.append(shoreline_gdf)
        counter += 1

        # Save the collected shorelines so far
        if counter % 3 == 0 and len(all_extracted_shorelines) > 0:
            shorelines_gdf = concat_and_sort_geodataframes(all_extracted_shorelines, "date")
            extracted_shorelines_path = os.path.join(directory, f"raw_extracted_shorelines.geojson")
            shorelines_gdf.to_file(extracted_shorelines_path, driver="GeoJSON")
            print(f"Saved extracted shorelines to {extracted_shorelines_path}")

    # put all the shorelines into a single geodataframe
    if len(all_extracted_shorelines) == 0:
        print(f"No shorelines extracted from {directory}. Double check the settings and the files in the directory.")
        return {'shorelines':[],'dates':[]}

    shorelines_gdf = concat_and_sort_geodataframes(all_extracted_shorelines, "date")

    # save the geodataframe to a geojson file
    extracted_shorelines_path = os.path.join(directory, f"raw_extracted_shorelines.geojson")  
    stringified_shoreline_gdf = stringify_datetime_columns(shorelines_gdf.copy())
    stringified_shoreline_gdf.to_file(extracted_shorelines_path, driver="GeoJSON")
    print(f"saved extracted shorelines to {extracted_shorelines_path}")

    # convert the geodataframe to a dictionary
    shorelines_dict = convert_shoreline_gdf_to_dict(shorelines_gdf,date_format='%Y-%m-%dT%H:%M:%S%z')
    return shorelines_dict

def extract_shorelines(directory:str,
                    suffix:str,
                    model_card_path:str,
                    extract_shorelines_settings:dict,
                    shoreline_buffer_tiff:str=None,
                    cloud_mask_suffix:str = '3B_udm2_clip.tif',
                    separator = '_3B'):
                       

    filtered_tiffs = glob.glob(os.path.join(directory, f"*{suffix}.tif"))
    if len(filtered_tiffs) == 0:
        print("No tiffs found in the directory")                  
                       
    # if length is 1 then call a single function to do all this stuff
    all_extracted_shorelines = []
    for target_path in tqdm(
            glob.glob(os.path.join(directory, f"*{suffix}.tif")),
            desc="extracting shorelines"
        ):

        if not os.path.exists(target_path):
            print(f"Skipping {target_path} because it does not exist")
            continue
        base_filename = processing.get_base_filename(target_path, separator)
        # get the processed coregistered file, the cloud mask and the npz file
        cloud_mask_path = utils.get_file_path(directory, base_filename, f"*{cloud_mask_suffix}")
        if not cloud_mask_path:
            print(f"Skipping {cloud_mask_path} because cloud mask not found")
            continue
        # get the date from the path
        date=processing.get_date_from_path(base_filename)

        # apply the model to the image
        model.apply_model_to_image(target_path,directory)
        # this is the file generated by the model
        npz_path = os.path.join(directory, 'out', f"{base_filename}{suffix}_res.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping {npz_path} because it does not exist")
            continue
            
        save_path = os.path.join(directory, 'shoreline_detection_figures')
        # this shoreline is in the UTM projection
        shoreline_gdf = get_shorelines_from_model(cloud_mask_path,target_path,model_card_path,npz_path,date,satname= 'planet', settings = extract_shorelines_settings,topobathy_path = shoreline_buffer_tiff,
                                            save_path=save_path)
        all_extracted_shorelines.append(shoreline_gdf)

    # put all the shorelines into a single geodataframe
    shorelines_gdf = concat_and_sort_geodataframes(all_extracted_shorelines, "date")

    # save the geodataframe to a geojson file
    extracted_shorelines_path = os.path.join(directory, f"raw_extracted_shorelines.geojson")    
    shorelines_gdf.to_file(extracted_shorelines_path, driver="GeoJSON")
    print(f"saved extracted shorelines to {extracted_shorelines_path}")

    # then intersect these shorelines with the transects
    shorelines_dict = convert_shoreline_gdf_to_dict(shorelines_gdf,date_format='%Y-%m-%dT%H:%M:%S%z')
    return shorelines_dict


def get_shorelines_from_model_reference_shoreline_gdf(planet_cloud_mask_path:str,
                              planet_path:str,
                              model_card_path,
                              npz_file,
                              date,
                              satname,
                              settings,
                              reference_shoreline_gdf:gpd.GeoDataFrame,
                              save_path=""):
    """
    Extracts shorelines from a model using the provided inputs.

    Args:
        planet_cloud_mask_path (str): The file path to the planet cloud mask.
        planet_path (str): The file path to the planet image.
        model_card_path: The file path to the model card.
        npz_file: The file path to the npz file.
        date: The date of the shoreline extraction.
        satname: The name of the satellite.
        settings: The settings for the shoreline extraction.
        reference_shoreline_gdf (gpd.GeoDataFrame): The reference shoreline as a GeoDataFrame.
        save_path (str, optional): The directory path to save the extracted shoreline. Defaults to the current working directory.

    Returns:
        geopandas.GeoDataFrame: The extracted shoreline as a GeoDataFrame.
    """

    if not save_path:
        save_path = os.path.dirname(planet_path)
    # get the labels for water and land
    water_classes_indices,land_mask,class_mapping  = get_class_indices_from_model_card(npz_file,model_card_path) 
    all_labels = load_image_labels(npz_file)
    # remove any segments of land that are too small to be considered beach from the land mask
    min_beach_area = settings.get("min_beach_area", 10000)
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)

    # ge the no data and cloud masks
    im_nodata = rasterio.open(planet_cloud_mask_path).read(8)
    cloud_mask = rasterio.open(planet_cloud_mask_path).read(6)
    cloud_shadow_mask = rasterio.open(planet_cloud_mask_path).read(3)
    # combine the cloud and shadow mask
    cloud_and_shadow_mask = np.logical_or(cloud_mask, cloud_shadow_mask)

    planet_RGB = read_planet_tiff(planet_path,[1,2,3])
    image_epsg = get_epsg_from_tiff(planet_path)
    if image_epsg is None:
        raise ValueError(f"The image does not have a valid EPSG code. Image : {planet_path}")
    image_epsg = int(image_epsg)
    georef = get_georef(planet_path)

    # for planet the pixel size is 3m and the land mask is the same size as the image
    # this creates a reference shoreline buffer in pixel coordinates
    utm_zone = get_utm_zone_from_geotiff(planet_path)
    reference_shoreline_tif_path = os.path.join(os.path.dirname(planet_path),'reference_shoreline.tiff')
    ref_sl_path = dilate_shoreline(reference_shoreline_gdf, buffer_size=100, output_file=reference_shoreline_tif_path, utm_zone=utm_zone)
    reference_shoreline_buffer = processing.get_mask_in_matching_projection(planet_path,ref_sl_path)

    # find the contours of the land mask & then filter out the shorelines that are too close to the cloud mask
    contours = simplified_find_contours(
        land_mask,
        reference_shoreline_buffer=reference_shoreline_buffer
        )
    # this shoreline is in UTM coordinates
    
    filtered_shoreline_gdf = process_shoreline(
            contours,
            cloud_and_shadow_mask,
            im_nodata,
            georef,
            image_epsg,
            settings,
            date,
        )
    # convert the shorelines to a list of numpy arrays that can be plotted
    single_shoreline = []
    for geom in filtered_shoreline_gdf.geometry:
        single_shoreline.append(np.array(geom.coords))
    shoreline = extract_contours(single_shoreline)

    shoreline_detection_figures(
        planet_RGB,
        cloud_and_shadow_mask,
        land_mask,
        all_labels,
        shoreline,
        image_epsg,
        georef,
        date,
        "planet",
        class_mapping,
        settings["output_epsg"],
        save_path=save_path,
        reference_shoreline_buffer=reference_shoreline_buffer,
    )
    return filtered_shoreline_gdf  

def get_shorelines_from_model_reference_shoreline(planet_cloud_mask_path:str,
                              planet_path:str,
                              model_card_path,
                              npz_file,
                              date,
                              satname,
                              settings,
                              reference_shoreline:np.ndarray,
                              save_path=""):
    """
    Extracts shorelines from a model using the provided inputs.

    Args:
        planet_cloud_mask_path (str): The file path to the planet cloud mask.
        planet_path (str): The file path to the planet image.
        model_card_path: The file path to the model card.
        npz_file: The file path to the npz file.
        date: The date of the shoreline extraction.
        satname: The name of the satellite.
        settings: The settings for the shoreline extraction.
        topobathy_path (str, optional): The file path to the topobathy mask. Defaults to None.
        save_path (str, optional): The directory path to save the extracted shoreline. Defaults to the current working directory.

    Returns:
        geopandas.GeoDataFrame: The extracted shoreline as a GeoDataFrame.
    """

    if not save_path:
        save_path = os.path.dirname(planet_path)
    # get the labels for water and land
    water_classes_indices,land_mask,class_mapping  = get_class_indices_from_model_card(npz_file,model_card_path) 
    all_labels = load_image_labels(npz_file)
    # remove any segments of land that are too small to be considered beach from the land mask
    min_beach_area = settings.get("min_beach_area", 10000)
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)

    # ge the no data and cloud masks
    im_nodata = rasterio.open(planet_cloud_mask_path).read(8)
    cloud_mask = rasterio.open(planet_cloud_mask_path).read(6)
    cloud_shadow_mask = rasterio.open(planet_cloud_mask_path).read(3)
    # combine the cloud and shadow mask
    cloud_and_shadow_mask = np.logical_or(cloud_mask, cloud_shadow_mask)

    planet_RGB = read_planet_tiff(planet_path,[1,2,3])
    image_epsg = get_epsg_from_tiff(planet_path)
    if image_epsg is None:
        raise ValueError(f"The image does not have a valid EPSG code. Image : {planet_path}")
    image_epsg = int(image_epsg)
    georef = get_georef(planet_path)

    # for planet the pixel size is 3m and the land mask is the same size as the image
    # this creates a reference shoreline buffer in pixel coordinates
    reference_shoreline_buffer = create_shoreline_buffer(land_mask.shape, georef, image_epsg, 3, reference_shoreline,settings)

    # find the contours of the land mask & then filter out the shorelines that are too close to the cloud mask
    contours = simplified_find_contours(
        land_mask,
        reference_shoreline_buffer=reference_shoreline_buffer
        )
    # this shoreline is in UTM coordinates
    
    filtered_shoreline_gdf = process_shoreline(
            contours,
            cloud_and_shadow_mask,
            im_nodata,
            georef,
            image_epsg,
            settings,
            date,
        )
    # convert the shorelines to a list of numpy arrays that can be plotted
    single_shoreline = []
    for geom in filtered_shoreline_gdf.geometry:
        single_shoreline.append(np.array(geom.coords))
    shoreline = extract_contours(single_shoreline)

    shoreline_detection_figures(
        planet_RGB,
        cloud_and_shadow_mask,
        land_mask,
        all_labels,
        shoreline,
        image_epsg,
        georef,
        date,
        "planet",
        class_mapping,
        settings["output_epsg"],
        save_path=save_path,
        reference_shoreline_buffer=reference_shoreline_buffer,
    )
    return filtered_shoreline_gdf               


def get_shorelines_from_model(planet_cloud_mask_path:str,
                              planet_path:str,
                              model_card_path,
                              npz_file,
                              date,
                              satname,
                              settings,
                              topobathy_path=None,
                              save_path=""):
    """
    Extracts shorelines from a model using the provided inputs.

    Args:
        planet_cloud_mask_path (str): The file path to the planet cloud mask.
        planet_path (str): The file path to the planet image.
        model_card_path: The file path to the model card.
        npz_file: The file path to the npz file.
        date: The date of the shoreline extraction.
        satname: The name of the satellite.
        settings: The settings for the shoreline extraction.
        topobathy_path (str, optional): The file path to the topobathy mask. Defaults to None.
        save_path (str, optional): The directory path to save the extracted shoreline. Defaults to the current working directory.

    Returns:
        geopandas.GeoDataFrame: The extracted shoreline as a GeoDataFrame.
    """

    if not save_path:
        save_path = os.path.dirname(planet_path)
    # get the labels for water and land
    water_classes_indices,land_mask,class_mapping  = get_class_indices_from_model_card(npz_file,model_card_path) 
    all_labels = load_image_labels(npz_file)
    # remove any segments of land that are too small to be considered beach from the land mask
    min_beach_area = 10000
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)
    # read the elevation msak if it exists
    mask_data = None

    # read the shoreline buffer from the topobathy mask
    if os.path.exists(topobathy_path):
        mask_data = processing.get_mask_in_matching_projection(planet_path,topobathy_path)

    # ge the no data and cloud masks
    im_nodata = rasterio.open(planet_cloud_mask_path).read(8)
    cloud_mask = rasterio.open(planet_cloud_mask_path).read(6)
    cloud_shadow_mask = rasterio.open(planet_cloud_mask_path).read(3)
    # combine the cloud and shadow mask
    cloud_and_shadow_mask = np.logical_or(cloud_mask, cloud_shadow_mask)

    planet_RGB = read_planet_tiff(planet_path,[1,2,3])
    image_epsg = get_epsg_from_tiff(planet_path)
    if image_epsg is None:
        raise ValueError(f"The image does not have a valid EPSG code. Image : {planet_path}")
    image_epsg = int(image_epsg)
    georef = get_georef(planet_path)

    # find the contours of the land mask & then filter out the shorelines that are too close to the cloud mask
    contours = simplified_find_contours(
        land_mask,
        reference_shoreline_buffer=mask_data
        )
    # this shoreline is in UTM coordinates
    filtered_shoreline_gdf = process_shoreline(
            contours,
            cloud_and_shadow_mask,
            im_nodata,
            georef,
            image_epsg,
            settings,
            date,
        )
    # convert the shorelines to a list of numpy arrays that can be plotted
    single_shoreline = []
    for geom in filtered_shoreline_gdf.geometry:
        single_shoreline.append(np.array(geom.coords))
    shoreline = extract_contours(single_shoreline)

    shoreline_detection_figures(
        planet_RGB,
        cloud_and_shadow_mask,
        land_mask,
        all_labels,
        shoreline,
        image_epsg,
        georef,
        date,
        "planet",
        class_mapping,
        settings["output_epsg"],
        save_path=save_path,
        reference_shoreline_buffer=mask_data,
    )
    return filtered_shoreline_gdf
    

def process_shoreline(
    contours, cloud_mask, im_nodata, georef, image_epsg, settings, date, **kwargs
):
    # convert the contours that are currently pixel coordinates to world coordiantes
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    contours_epsg = SDS_tools.convert_epsg(
        contours_world, image_epsg, settings["output_epsg"]
    )
    # this is the shoreline in the form of a list of numpy arrays, each array containing the coordinates of a shoreline x,y,z
    contours_long = filter_contours_by_length(contours_epsg, settings["min_length_sl"])
    # this removes the z coordinate from each shoreline point, so the format is list of numpy arrays, each array containing the x,y coordinates of a shoreline point
    contours_2d = [contour[:, :2] for contour in contours_long]
    # remove shoreline points that are too close to the no data mask
    new_contours = filter_points_within_distance_to_mask(
        contours_2d,
        im_nodata,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=60,
    )
    # remove shoreline points that are too close to the cloud mask
    new_contours = filter_points_within_distance_to_mask(
        new_contours,
        cloud_mask,
        georef,
        image_epsg,
        settings["output_epsg"],
        distance_threshold=settings["dist_clouds"],
    )
    filtered_contours_long = filter_contours_by_length(
        new_contours, settings["min_length_sl"]
    )
    contours_shapely = [LineString(contour) for contour in filtered_contours_long]
    date_obj = datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
    gdf = gpd.GeoDataFrame(
        {"date": np.tile(date_obj, len(contours_shapely))},
        geometry=contours_shapely,
        crs=f"EPSG:{image_epsg}",
    )
    return gdf

def filter_contours_by_length(contours_epsg: list[np.ndarray], min_length_sl: float) -> list[np.ndarray]:
    """
    Filters contours by their length.

    Args:
        contours_epsg (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        min_length_sl (float): Minimum length threshold for the contours.

    Returns:
        list[np.ndarray]: List of contours that meet the minimum length requirement.
    """
    contours_long = []
    for wl in contours_epsg:
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        if len(coords) < 2:
            continue
        a = LineString(coords)
        if a.length >= min_length_sl:
            contours_long.append(wl)
    return contours_long

def filter_points_within_distance_to_mask(contours_2d: list[np.ndarray], mask: np.ndarray, georef: np.ndarray,
                                          image_epsg: int, output_epsg: int, distance_threshold: float = 60) -> list[np.ndarray]:
    """
    Filters points within a specified distance to a mask.

    Args:
        contours_2d (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        mask (np.ndarray): Binary mask array.
        georef (np.ndarray): Georeference information.
        image_epsg (int): EPSG code of the image coordinate system.
        output_epsg (int): EPSG code of the output coordinate system.
        distance_threshold (float, optional): Distance threshold for filtering. Defaults to 60.

    Returns:
        list[np.ndarray]: List of contours filtered by the distance to the mask.
    """
    idx_mask = np.where(mask)
    idx_mask = np.array(
        [(idx_mask[0][k], idx_mask[1][k]) for k in range(len(idx_mask[0]))]
    )
    if len(idx_mask) == 0:
        return contours_2d
    coords_in_epsg = SDS_tools.convert_epsg(
        SDS_tools.convert_pix2world(idx_mask, georef), image_epsg, output_epsg
    )[:, :-1]
    coords_tree = KDTree(coords_in_epsg)
    new_contours = filter_shorelines_by_distance(
        contours_2d, coords_tree, distance_threshold
    )
    return new_contours

def convert_world2pix(
    points: list[np.ndarray] | np.ndarray, georef: np.ndarray
) -> list[np.ndarray] | np.ndarray:
    """
    Converts world coordinates to pixel coordinates.

    Args:
        points (list[np.ndarray] | np.ndarray): List of points or array of points in world coordinates.
        georef (np.ndarray): Georeference information.

    Returns:
        list[np.ndarray] | np.ndarray: Converted points in pixel coordinates.
    """
    aff_mat = np.array(
        [
            [georef[1], georef[2], georef[0]],
            [georef[4], georef[5], georef[3]],
            [0, 0, 1],
        ]
    )
    tform = transform.AffineTransform(aff_mat)
    if isinstance(points, list):
        points_converted = [
            tform.inverse(arr[:, :2]) if arr.ndim == 2 else tform.inverse(arr)
            for arr in points
        ]
    elif isinstance(points, np.ndarray):
        points_converted = tform.inverse(points)
    else:
        raise ValueError("Invalid input type")
    return points_converted

def filter_shorelines_by_distance(
    contours_2d: list[np.ndarray], coords_tree: KDTree, distance_threshold: float = 60
) -> list[np.ndarray]:
    """
    Filters shorelines by their distance to a set of coordinates.

    Args:
        contours_2d (list[np.ndarray]): List of contours, where each contour is an array of coordinates.
        coords_tree (KDTree): KDTree of coordinates to compare distances against.
        distance_threshold (float, optional): Distance threshold for filtering. Defaults to 60.

    Returns:
        list[np.ndarray]: List of filtered shorelines.
    """
    new_contours = []
    for shoreline in contours_2d:
        distances, _ = coords_tree.query(
            shoreline, distance_upper_bound=distance_threshold
        )
        idx_keep = distances >= distance_threshold
        new_shoreline = shoreline[idx_keep]
        if len(new_shoreline) > 0:
            new_contours.append(new_shoreline)
    return new_contours

def concat_and_sort_geodataframes(
    gdfs: list[gpd.GeoDataFrame], date_column: str
) -> gpd.GeoDataFrame:
    """
    Concatenates a list of GeoDataFrames with the same columns into a single GeoDataFrame and sorts by a date column.

    Args:
        gdfs (list[gpd.GeoDataFrame]): List of GeoDataFrames to concatenate.
        date_column (str): The name of the date column to sort by.

    Returns:
        gpd.GeoDataFrame: A single concatenated and sorted GeoDataFrame.
    """
    concatenated_gdf = pd.concat(gdfs, ignore_index=True)
    concatenated_gdf = gpd.GeoDataFrame(concatenated_gdf)
    concatenated_gdf[date_column] = pd.to_datetime(
        concatenated_gdf[date_column]
    )  # Ensure the date column is in datetime format

    sorted_gdf = concatenated_gdf.sort_values(by=date_column).reset_index(drop=True)
    # convert the date to a string

    return sorted_gdf


def shoreline_detection_figures(
    im_ms: np.ndarray,
    cloud_mask: np.ndarray,
    merged_labels: np.ndarray,
    all_labels: np.ndarray,
    shoreline: np.ndarray,
    image_epsg: str,
    georef,
    date: str,
    satname: str,
    class_mapping: dict,
    output_epsg: int ,
    save_path: str,
    reference_shoreline_buffer: np.ndarray = None,
):
    # over the merged labels from true false to 1 0
    merged_labels = merged_labels.astype(int)


    # Convert shoreline points to pixel coordinates
    try:
        pixelated_shoreline =convert_world2pix(
            SDS_tools.convert_epsg(shoreline, output_epsg, image_epsg)[
                :, [0, 1]
            ],
            georef,
        )
    except:
        pixelated_shoreline = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    # Create legend for the shorelines
    black_line = mlines.Line2D([], [], color="k", linestyle="-", label="shoreline")
    # The additional patches to be appended to the legend
    additional_legend_items = [black_line]

    if reference_shoreline_buffer is not None:
        buffer_patch = mpatches.Patch(
        color=(0.1450980392156863, 0.8588235294117647, 0.33725490196078434, 1.0), alpha=0.80, label="Reference shoreline buffer"
        )
        additional_legend_items.append(buffer_patch)

    
    # create a legend for the class colors and the shoreline
    all_class_color_mapping = plotting.create_class_color_mapping(all_labels)
    # the class mapping has the labels as str instead of int so we need to convert them to int
    class_mapping = {int(k):v for k,v in class_mapping.items()}
    all_classes_legend = create_legend(
        class_mapping,all_class_color_mapping, additional_patches=additional_legend_items
    )

    class_color_mapping = plotting.create_class_color_mapping(merged_labels)
    merged_classes_legend = create_legend(
        class_mapping={0: "other", 1: "water"},
        color_mapping=class_color_mapping,
        additional_patches=additional_legend_items,
    )

    class_color_mapping = plotting.create_class_color_mapping(merged_labels)
    fig = plot_image_with_legend(im_ms, 
                                 merged_labels,
                                 all_labels,
                                 pixelated_shoreline,
                                 merged_classes_legend,
                                 all_classes_legend,
                                 class_color_mapping,
                                 all_class_color_mapping,
                                 reference_shoreline_buffer=reference_shoreline_buffer,
                                 titles=['Original Image', 'Merged Classes', 'All Classes'],)

    # save a .jpg under /jpg_files/detection
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    save_detection_figure(fig, save_path, date, satname)
    plt.close(fig)
    
# under active development
# I want to improve this functions speed and have it remove shorelines whose length is less than min shoreline length after the contours are found
# to do this I need to figure out who to get around the shorelines being converted to numpy arrays of just x in one and y in the other because the individual shorelines & how they are connected are lost
# def process_shoreline(
#     contours, cloud_mask,  georef, image_epsg, settings, **kwargs
# ):
#     # convert pixel coordinates to world coordinates
#     contours_world = SDS_tools.convert_pix2world(contours, georef)
#     # convert world coordinates to desired spatial reference system
#     contours_epsg = SDS_tools.convert_epsg(
#         contours_world, image_epsg, settings["output_epsg"]
#     )

#     # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
#     # this enables to remove the very small contours that do not correspond to the shoreline
#     contours_long = []
#     for l, wl in enumerate(contours_epsg):
#         coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
#         a = LineString(coords)  # shapely LineString structure
#         # print(a.length)
#         if a.length >= settings["min_length_sl"]:
#             contours_long.append(wl)
#     # format points into np.array
#     x_points = np.array([])
#     y_points = np.array([])
#     for k in range(len(contours_long)):
#         x_points = np.append(x_points, contours_long[k][:, 0])
#         y_points = np.append(y_points, contours_long[k][:, 1])
#     contours_array = np.transpose(np.array([x_points, y_points]))

#     shoreline = contours_array


#     print(
#         f"Number of shorelines before removing shorelines < {settings['min_length_sl']}m: {len(contours_epsg)} shorelines. Number of shorelines after filtering shorelines: {len(contours_long)} shorelines"
#     )

#     if len(shoreline) == 0:
#         return shoreline

#     # # 2. Remove any shoreline points that are close to cloud pixels (effect of shadows)
#     # if np.sum(np.sum(cloud_mask)) > 0:
#     #     # get the coordinates of the cloud pixels
#     #     idx_cloud = np.where(cloud_mask)
#     #     idx_cloud = np.array(
#     #         [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
#     #     )
#     #     # convert to world coordinates and same epsg as the shoreline points
#     #     coords_cloud = SDS_tools.convert_epsg(
#     #         SDS_tools.convert_pix2world(idx_cloud, georef),
#     #         image_epsg,
#     #         settings["output_epsg"],
#     #     )[:, :-1]
#     #     # only keep the shoreline points that are at least 30m from any cloud pixel
#     #     idx_keep = np.ones(len(shoreline)).astype(bool)
#     #     for k in range(len(shoreline)):
#     #         if np.any(
#     #             np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1)
#     #             < settings["dist_clouds"]
#     #         ):
#     #             idx_keep[k] = False
#     #     shoreline = shoreline[idx_keep]
    
#     # if np.sum(np.sum(no_data)) > 0:
#     #     # get the coordinates of the cloud pixels
#     #     idx_cloud = np.where(no_data)
#     #     idx_cloud = np.array(
#     #         [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
#     #     )
#     #     # convert to world coordinates and same epsg as the shoreline points
#     #     coords_cloud = SDS_tools.convert_epsg(
#     #         SDS_tools.convert_pix2world(idx_cloud, georef),
#     #         image_epsg,
#     #         settings["output_epsg"],
#     #     )[:, :-1]
#     #     # only keep the shoreline points that are at least 30m from any nodata pixel
#     #     idx_keep = np.ones(len(shoreline)).astype(bool)
#     #     for k in range(len(shoreline)):
#     #         if np.any(np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1) < 30):
#     #             idx_keep[k] = False
#     #     shoreline = shoreline[idx_keep]


#     return shoreline

def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact
    with clouds.

    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours

    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs)

    """

    # Remove contours that contain NaNs
    contours_nonans = [
        contour[~np.isnan(contour).any(axis=1)] for contour in contours
        if not np.isnan(contour).all(axis=1).any()
    ]

    # Filter out empty contours
    contours_nonans = [contour for contour in contours_nonans if len(contour) > 1]

    return contours_nonans

def simplified_find_contours(
    im_labels: np.array, cloud_mask: np.array=None, reference_shoreline_buffer: np.array=None
) -> List[np.array]:
    """Find contours in a binary image using skimage.measure.find_contours and processes out contours that contain NaNs.
    Parameters:
    -----------
    im_labels: np.nd.array
        binary image with 0s and 1s
    cloud_mask: np.array
        boolean array indicating cloud mask
    Returns:
    -----------
    processed_contours: list of arrays
        processed image contours (only the ones that do not contains NaNs)
    """
    # make a copy of the im_labels array as a float (this allows find contours to work))
    im_labels_masked = im_labels.copy().astype(float)
    # Apply the cloud mask by setting masked pixels to NaN
    if cloud_mask is not None:
        im_labels_masked[cloud_mask] = np.NaN
    # only keep the pixels inside the reference shoreline buffer
    if reference_shoreline_buffer is not None:
        reference_shoreline_buffer = reference_shoreline_buffer[:im_labels.shape[0],:im_labels.shape[1]]
        im_labels_masked[~reference_shoreline_buffer] = np.NaN
    
    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels_masked, 0.5)

    # remove contour points that are NaNs (around clouds and nodata intersections)
    processed_contours = process_contours(contours)

    return processed_contours
