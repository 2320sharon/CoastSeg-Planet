import numpy as np
import skimage.measure as measure
from typing import List
from shapely.geometry import LineString

from coastseg_planet.plotting import create_overlay, create_legend, plot_image_with_legend, save_detection_figure, create_class_color_mapping
from coastseg_planet.processing import read_planet_tiff, get_georef, get_epsg_from_tiff, create_gdf_from_shoreline
from coastsat import SDS_tools

from coastseg.extracted_shoreline import load_image_labels, load_merged_image_labels, remove_small_objects_and_binarize, get_indices_of_classnames, get_class_mapping
from coastseg import file_utilities

import os
import matplotlib.pyplot as plt
from matplotlib import patches, lines
import colorsys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from  coastsat.SDS_preprocess import rescale_image_intensity



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
    
def get_shorelines_from_model(planet_cloud_mask_path:str,planet_path:str,model_card_path,npz_file,date,satname,settings,save_path=os.getcwd()):

    # get the labels for water and land
    water_classes_indices,land_mask,class_mapping  = get_class_indices_from_model_card(npz_file,model_card_path)
    
    all_labels = load_image_labels(npz_file)
    # remove any segments of land that are too small to be considered beach from the land mask
    min_beach_area = 10000
    land_mask = remove_small_objects_and_binarize(land_mask, min_beach_area)
    # find the contours of the land mask
    contours = simplified_find_contours(
        land_mask
        )
    planet_cloud_mask = read_planet_tiff(planet_cloud_mask_path,[1])
    planet_RGB = read_planet_tiff(planet_path,[1,2,3])
    image_epsg = get_epsg_from_tiff(planet_path)
    if image_epsg is None:
        raise ValueError(f"The image does not have a valid EPSG code. Image : {planet_path}")
    image_epsg = int(image_epsg)
    print(f"image_epsg: {image_epsg} type: {type(image_epsg)}")
    georef = get_georef(planet_path)
    shoreline = process_shoreline(contours,planet_cloud_mask,  georef, image_epsg, settings)
    # save the shoreline to a geojson file
    shoreline_gdf = create_gdf_from_shoreline(shoreline,settings["output_epsg"],"lines")
    shoreline_gdf.to_crs(epsg=4326, inplace=True)
    shoreline_geojson_path = os.path.join(save_path,f"extracted_shoreline_{date}.geojson")
    shoreline_gdf.to_file(shoreline_geojson_path, driver="GeoJSON")

    print(f"class_mapping: {class_mapping}")
    shoreline_detection_figures(
        planet_RGB,
        planet_cloud_mask,
        land_mask,
        all_labels,
        shoreline,
        image_epsg,
        georef,
        date,
        satname,
        class_mapping,
        settings["output_epsg"],
        save_path
    )
    return shoreline
    
    
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
):

    print(f"class_mapping: {class_mapping}")
    # over the merged labels from true false to 1 0
    merged_labels = merged_labels.astype(int)


    # Convert shoreline points to pixel coordinates
    try:
        pixelated_shoreline = SDS_tools.convert_world2pix(
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
    
    # create a legend for the class colors and the shoreline
    all_class_color_mapping = create_class_color_mapping(all_labels)
    # the class mapping has the labels as str instead of int so we need to convert them to int
    class_mapping = {int(k):v for k,v in class_mapping.items()}
    all_classes_legend = create_legend(
        class_mapping,all_class_color_mapping, additional_patches=additional_legend_items
    )

    additional_legend_items = [black_line]
    class_color_mapping = create_class_color_mapping(merged_labels)
    merged_classes_legend = create_legend(
        class_mapping={0: "other", 1: "water"},
        color_mapping=class_color_mapping,
        additional_patches=additional_legend_items,
    )
    print(f"all_class_color_mapping: {all_class_color_mapping}")

    class_color_mapping = create_class_color_mapping(merged_labels)
    fig = plot_image_with_legend(im_ms, merged_labels,all_labels,
                                 pixelated_shoreline,
                                 merged_classes_legend,all_classes_legend,class_color_mapping,all_class_color_mapping,titles=['Original Image', 'Merged Classes', 'All Classes'],)

    # save a .jpg under /jpg_files/detection
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    save_detection_figure(fig, save_path, date, satname)
    plt.close(fig)
    
# under active development
# I want to improve this functions speed and have it remove shorelines whose length is less than min shoreline length after the contours are found
# to do this I need to figure out who to get around the shorelines being converted to numpy arrays of just x in one and y in the other because the individual shorelines & how they are connected are lost
def process_shoreline(
    contours, cloud_mask,  georef, image_epsg, settings, **kwargs
):
    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(
        contours_world, image_epsg, settings["output_epsg"]
    )

    # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        print(a.length)
        if a.length >= settings["min_length_sl"]:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    contours_array = np.transpose(np.array([x_points, y_points]))

    shoreline = contours_array


    print(
        f"Number of shorelines before removing shorelines < {settings['min_length_sl']}m: {len(contours_epsg)} shorelines. Number of shorelines after filtering shorelines: {len(contours_long)} shorelines"
    )

    if len(shoreline) == 0:
        return shoreline

    # # 2. Remove any shoreline points that are close to cloud pixels (effect of shadows)
    # if np.sum(np.sum(cloud_mask)) > 0:
    #     # get the coordinates of the cloud pixels
    #     idx_cloud = np.where(cloud_mask)
    #     idx_cloud = np.array(
    #         [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
    #     )
    #     # convert to world coordinates and same epsg as the shoreline points
    #     coords_cloud = SDS_tools.convert_epsg(
    #         SDS_tools.convert_pix2world(idx_cloud, georef),
    #         image_epsg,
    #         settings["output_epsg"],
    #     )[:, :-1]
    #     # only keep the shoreline points that are at least 30m from any cloud pixel
    #     idx_keep = np.ones(len(shoreline)).astype(bool)
    #     for k in range(len(shoreline)):
    #         if np.any(
    #             np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1)
    #             < settings["dist_clouds"]
    #         ):
    #             idx_keep[k] = False
    #     shoreline = shoreline[idx_keep]
    
    # if np.sum(np.sum(no_data)) > 0:
    #     # get the coordinates of the cloud pixels
    #     idx_cloud = np.where(no_data)
    #     idx_cloud = np.array(
    #         [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
    #     )
    #     # convert to world coordinates and same epsg as the shoreline points
    #     coords_cloud = SDS_tools.convert_epsg(
    #         SDS_tools.convert_pix2world(idx_cloud, georef),
    #         image_epsg,
    #         settings["output_epsg"],
    #     )[:, :-1]
    #     # only keep the shoreline points that are at least 30m from any nodata pixel
    #     idx_keep = np.ones(len(shoreline)).astype(bool)
    #     for k in range(len(shoreline)):
    #         if np.any(np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1) < 30):
    #             idx_keep[k] = False
    #     shoreline = shoreline[idx_keep]


    return shoreline

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
    if cloud_mask:
        im_labels_masked[cloud_mask] = np.NaN
    # only keep the pixels inside the reference shoreline buffer
    if reference_shoreline_buffer:
        im_labels_masked[~reference_shoreline_buffer] = np.NaN
    
    # 0 or 1 labels means 0.5 is the threshold
    contours = measure.find_contours(im_labels_masked, 0.5)

    # remove contour points that are NaNs (around clouds and nodata intersections)
    processed_contours = process_contours(contours)

    return processed_contours
