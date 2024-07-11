import numpy as np
import rasterio
import pandas as pd
import skimage.measure as measure
import geopandas as gpd
import glob
from typing import List
from shapely.geometry import LineString

from coastseg_planet.plotting import create_overlay, create_legend, plot_image_with_legend, save_detection_figure, create_class_color_mapping
from coastseg_planet.processing import read_planet_tiff, get_georef, get_epsg_from_tiff, create_gdf_from_shoreline
from coastseg_planet import processing 
from coastseg_planet import model
from coastseg_planet import transects
from coastseg_planet import utils
from coastsat import SDS_tools

from coastseg.extracted_shoreline import  load_merged_image_labels, remove_small_objects_and_binarize, get_indices_of_classnames, get_class_mapping
from coastseg import file_utilities

import os
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import colorsys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from scipy.spatial import KDTree
from shapely.geometry import LineString
from skimage import transform
from datetime import datetime

from  coastsat.SDS_preprocess import rescale_image_intensity



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

# def extract_shorelines():
#     npz_file = r'C:\development\coastseg-planet\20230412_094432_88_2439_3B_coregistered_model_ready_res.npz'
#     model_card_path = r'C:\development\coastseg-planet\CoastSeg-Planet\output_zoo\coastseg_planet\coastseg_planet\config\model_card.yml'

#     weights_directory = r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\downloaded_models\segformer_RGB_4class_8190958'

#     model_card_path = file_utilities.find_file_by_regex(
#         weights_directory, r".*modelcard\.json$"
#     )
#     settings = {
#     'output_epsg': 32631,
#     'min_length_sl': 100,
#     'dist_clouds': 50,
#     }
#     get_shorelines_from_model(model_card_path,npz_file)
    
    
    
def get_class_indices_from_model_card(npz_file,model_card_path):
    # get the water index from the model card
    water_classes_indices = get_indices_of_classnames(
        model_card_path, ["water", "whitewater"]
    )
    # Sample class mapping {0:'water',  1:'whitewater', 2:'sand', 3:'rock'}
    class_mapping = get_class_mapping(model_card_path)

    # get the labels for water and land
    land_mask = load_merged_image_labels(npz_file, class_indices=water_classes_indices)
    
    return water_classes_indices,land_mask,class_mapping 

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
    # for target_path in filtered_tiffs:
    for target_path in (
            glob.glob(os.path.join(directory, f"*{suffix}.tif"))
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
        model.apply_model_to_image(target_path,directory,False,False)
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
    shorelines_dict = transects.convert_shoreline_gdf_to_dict(shorelines_gdf)
    return shorelines_dict
                       

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
    

# def get_shorelines_from_model(planet_cloud_mask_path:str,
#                               planet_path:str,
#                               model_card_path,
#                               npz_file,
#                               date,
#                               satname,
#                               settings,
#                               topobathy_path=None,
#                               save_path=os.getcwd()):
#     """
#     Extracts shorelines from a model using the provided inputs.

#     Args:
#         planet_cloud_mask_path (str): The file path to the planet cloud mask.
#         planet_path (str): The file path to the planet image.
#         model_card_path: The file path to the model card.
#         npz_file: The file path to the npz file.
#         date: The date of the shoreline extraction.
#         satname: The name of the satellite.
#         settings: The settings for the shoreline extraction.
#         topobathy_path (str, optional): The file path to the topobathy mask. Defaults to None.
#         save_path (str, optional): The directory path to save the extracted shoreline. Defaults to the current working directory.

#     Returns:
#         geopandas.GeoDataFrame: The extracted shoreline as a GeoDataFrame.
#     """
#     # get the labels for water and land
#     water_classes_indices,land_mask,class_mapping  = get_class_indices_from_model_card(npz_file,model_card_path) 
#     all_labels = load_image_labels(npz_file)
#     # remove any segments of land that are too small to be considered beach from the land mask
#     min_beach_area = 10000
#     land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)
#     # read the elevation msak if it exists
#     mask_data = None
#     if os.path.exists(topobathy_path):
#         mask_data = processing.read_topobathy_mask(topobathy_path)
    
#     # find the contours of the land mask
#     contours = simplified_find_contours(
#         land_mask,
#         reference_shoreline_buffer=mask_data
#         )
#     planet_cloud_mask = read_planet_tiff(planet_cloud_mask_path,[1])
#     planet_RGB = read_planet_tiff(planet_path,[1,2,3])
#     image_epsg = get_epsg_from_tiff(planet_path)
#     if image_epsg is None:
#         raise ValueError(f"The image does not have a valid EPSG code. Image : {planet_path}")
#     image_epsg = int(image_epsg)
#     # print(f"image_epsg: {image_epsg} type: {type(image_epsg)}")
#     georef = get_georef(planet_path)
#     shoreline = process_shoreline(contours,planet_cloud_mask,  georef, image_epsg, settings)
#     # save the shoreline to a geojson file
#     shoreline_gdf = create_gdf_from_shoreline(shoreline,settings["output_epsg"],date,"lines")
#     shoreline_gdf_epsg4326 = shoreline_gdf.to_crs(epsg=4326)
#     shoreline_geojson_path = os.path.join(save_path,f"extracted_shoreline_{date}.geojson")
#     shoreline_gdf_epsg4326.to_file(shoreline_geojson_path, driver="GeoJSON")

#     shoreline_detection_figures(
#         planet_RGB,
#         planet_cloud_mask,
#         land_mask,
#         all_labels,
#         shoreline,
#         image_epsg,
#         georef,
#         date,
#         satname,
#         class_mapping,
#         settings["output_epsg"],
#         save_path,
#         reference_shoreline_buffer=mask_data
#     )
#     return shoreline_gdf

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
    all_class_color_mapping = create_class_color_mapping(all_labels)
    # the class mapping has the labels as str instead of int so we need to convert them to int
    class_mapping = {int(k):v for k,v in class_mapping.items()}
    all_classes_legend = create_legend(
        class_mapping,all_class_color_mapping, additional_patches=additional_legend_items
    )

    class_color_mapping = create_class_color_mapping(merged_labels)
    merged_classes_legend = create_legend(
        class_mapping={0: "other", 1: "water"},
        color_mapping=class_color_mapping,
        additional_patches=additional_legend_items,
    )

    class_color_mapping = create_class_color_mapping(merged_labels)
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
