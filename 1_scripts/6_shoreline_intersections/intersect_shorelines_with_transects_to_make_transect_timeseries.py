import pandas as pd
import glob
import geopandas as gpd
import os
import numpy as np
import datetime
from tqdm import tqdm
from datetime import datetime
from shapely.geometry import MultiLineString,MultiPoint


def get_raw_timeseries(extracted_shorelines:dict, cross_distance_transects:dict):
    """
    Calculates the raw timeseries data by combining the extracted shorelines and cross-distance transects.


    Parameters:

    extracted_shorelines (dict): A dictionary containing the extracted shorelines.
        Must contain a 'dates' key with a list of dates and a 'shorelines' key with a list of numpy arrays of coordinates.
    cross_distance_transects (dict): A dictionary containing the cross-distance transects.
        The keys are transect names and the values are lists of cross distances.

    Returns:
    cross_distance_df (DataFrame): The raw timeseries data as a pandas DataFrame.
    """
    cross_distance_df = get_cross_distance_df(
        extracted_shorelines, cross_distance_transects
    )
    cross_distance_df.dropna(axis="columns", how="all", inplace=True)
    return cross_distance_df

def save_raw_timesseries(shorelines_dict, intersections_dict, save_location,filename ="raw_transect_time_series.csv",  verbose=True):
    """
    Save the raw transect time series to a CSV file.

    Parameters:
    - shorelines_dict (dict): A dictionary containing shorelines data.
    - intersections_dict (dict): A dictionary containing intersections data.
    - save_location (str): The directory where the CSV file will be saved.
    - verbose (bool): If True, print a message indicating the file path where the raw timeseries is saved.

    Returns:
    None
    """
    raw_timeseries = get_raw_timeseries(shorelines_dict, intersections_dict)

    # add .csv to end of filename if not already there
    if not filename.endswith(".csv"):
        # remove any extenstions from the filename
        filename = os.path.splitext(filename)[0]
        filename = filename + ".csv"

    filepath = os.path.join(save_location, filename)
    raw_timeseries.to_csv(filepath, sep=",", index=False)
    if verbose:
        print(f"Raw timeseries saved to {filepath}")

def get_cross_distance_df(
    extracted_shorelines: dict, cross_distance_transects: dict
) -> pd.DataFrame:
    """
    Creates a DataFrame from extracted shorelines and cross distance transects by
    getting the dates from extracted shorelines and saving it to the as the intersection time for each extracted shoreline
    for each transect

    Parameters:
    extracted_shorelines : dict
        A dictionary containing the extracted shorelines. It must have a "dates" key with a list of dates.
    cross_distance_transects : dict
        A dictionary containing the transects and the cross distance where the extracted shorelines intersected it. The keys are transect names and the values are lists of cross distances.
        eg.
        {  'tranect 1': [1,2,3],
            'tranect 2': [4,5,6],
        }
    Returns:
    DataFrame
        A DataFrame where each column is a transect from cross_distance_transects and the "dates" column from extracted_shorelines. Each row corresponds to a date and contains the cross distances for each transect on that date.
    """
    transects_csv = {}
    # copy dates from extracted shoreline
    transects_csv["dates"] = extracted_shorelines["dates"]
    # add cross distances for each transect within the ROI
    transects_csv = {**transects_csv, **cross_distance_transects}
    return pd.DataFrame(transects_csv)

def convert_shoreline_gdf_to_dict(shoreline_gdf, date_format="%d-%m-%Y", output_crs=None):
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

    for idx, row in shoreline_gdf.iterrows():
        date_str = row.date.strftime(date_format)
        geometry = row.geometry
        if geometry is not None:
            if isinstance(geometry,MultiPoint):
                dates.append(date_str) # only append the date once for the entire multipoint
                points_list = []
                shorelines_array = []
                for point in geometry.geoms:
                    points_list.append(list(point.coords[0]))
                # put all the points into a single numpy array to represent the shoreline for that date then append to shorelines
                shorelines_array = np.array(points_list)
                shorelines.append(shorelines_array)
            elif isinstance(geometry, MultiLineString):
                dates.append(date_str) # only append the date once for the entire multiline
                for line in geometry.geoms:
                    shorelines_array = np.array(line.coords)
                    shorelines.append(shorelines_array)
            else:
                shorelines_array = np.array(geometry.coords)
                shorelines.append(shorelines_array)
                dates.append(date_str)

    shorelines_dict = {'dates': dates, 'shorelines': shorelines}
    return shorelines_dict

def compute_intersection_QC(output, 
                            transects,
                            along_dist = 25, 
                            min_points = 3,
                            max_std = 15,
                            max_range = 30, 
                            min_chainage = -100, 
                            multiple_inter ="auto",
                            prc_multiple=0.1, 
                            use_progress_bar: bool = True):
    """
    
    More advanced function to compute the intersection between the 2D mapped shorelines
    and the transects. Produces more quality-controlled time-series of shoreline change.

    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        along_dist: int (in metres)
            alongshore distance to calculate the intersection (median of points
            within this distance).
        min_points: int
            minimum number of shoreline points to calculate an intersection.
        max_std: int (in metres)
            maximum std for the shoreline points when calculating the median,
            if above this value then NaN is returned for the intersection.
        max_range: int (in metres)
            maximum range for the shoreline points when calculating the median,
            if above this value then NaN is returned for the intersection.
        min_chainage: int (in metres)
            furthest landward of the transect origin that an intersection is
            accepted, beyond this point a NaN is returned.
        multiple_inter: mode for removing outliers ('auto', 'nan', 'max').
        prc_multiple: float, optional
            percentage to use in 'auto' mode to switch from 'nan' to 'max'.
        use_progress_bar(bool,optional). Defaults to True. If true uses tqdm to display the progress for iterating through transects.
            False, means no progress bar is displayed.

    Returns:
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally
            corrected.
    """

    # initialise dictionary with intersections for each transect
    cross_dist = dict([])

    shorelines = output["shorelines"]

    # loop through each transect
    transect_keys = transects.keys()
    if use_progress_bar:
        transect_keys = tqdm(
            transect_keys, desc="Computing transect shoreline intersections"
        )

    for key in transect_keys:
        # initialise variables
        std_intersect = np.zeros(len(shorelines))
        med_intersect = np.zeros(len(shorelines))
        max_intersect = np.zeros(len(shorelines))
        min_intersect = np.zeros(len(shorelines))
        n_intersect = np.zeros(len(shorelines))

        # loop through each shoreline
        for i in range(len(shorelines)):
            sl = shorelines[i]

            # in case there are no shoreline points
            if len(sl) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
                continue

            # compute rotation matrix
            X0 = transects[key][0, 0]
            Y0 = transects[key][0, 1]
            temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0, Y0])
            p2 = transects[key][-1, :]
            d_line = np.abs(np.cross(p2 - p1, sl - p1) / np.linalg.norm(p2 - p1))
            # calculate the distance between shoreline points and the origin of the transect
            d_origin = np.array([np.linalg.norm(sl[k, :] - p1) for k in range(len(sl))])
            # find the shoreline points that are close to the transects and to the origin
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            idx_close = np.where(idx_dist)[0]

            # in case there are no shoreline points close to the transect
            if len(idx_close) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
                    np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
                )
                xy_rot = np.matmul(Mrot, xy_close)
                # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
                xy_rot[0, xy_rot[0, :] < min_chainage] = np.nan

                # compute std, median, max, min of the intersections
                if not np.all(np.isnan(xy_rot[0, :])):
                    std_intersect[i] = np.nanstd(xy_rot[0, :])
                    med_intersect[i] = np.nanmedian(xy_rot[0, :])
                    max_intersect[i] = np.nanmax(xy_rot[0, :])
                    min_intersect[i] = np.nanmin(xy_rot[0, :])
                    n_intersect[i] = len(xy_rot[0, :])
                else:
                    std_intersect[i] = np.nan
                    med_intersect[i] = np.nan
                    max_intersect[i] = np.nan
                    min_intersect[i] = np.nan
                    n_intersect[i] = 0

        # quality control the intersections using dispersion metrics (std and range)
        condition1 = std_intersect <= max_std
        condition2 = (max_intersect - min_intersect) <= max_range
        condition3 = n_intersect >= min_points
        idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)

        # decide what to do with the intersections with high dispersion
        if multiple_inter == "auto":
            # compute the percentage of data points where the std is larger than the user-defined max
            prc_over = np.sum(std_intersect > max_std) / len(std_intersect)
            # if more than a certain percentage is above, use the maximum intersection
            if prc_over > prc_multiple:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            # otherwise put a nan
            else:
                med_intersect[~idx_good] = np.nan

        elif multiple_inter == "max":
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan

        elif multiple_inter == "nan":
            med_intersect[~idx_good] = np.nan

        else:
            raise Exception(
                "The multiple_inter parameter can only be: nan, max, or auto."
            )

        # store in dict
        cross_dist[key] = med_intersect

    return cross_dist

def get_transect_points_dict(feature: gpd.GeoDataFrame,identifier_column:str="id") -> dict:
    """Returns dict of np.arrays of transect start and end points
    Example
    {
        'usa_CA_0289-0055-NA1': array([[-13820440.53165404,   4995568.65036405],
        [-13820940.93156407,   4995745.1518021 ]]),
        'usa_CA_0289-0056-NA1': array([[-13820394.24579453,   4995700.97802925],
        [-13820900.16320004,   4995862.31860808]])
    }
    Args:
        feature (gpd.GeoDataFrame): clipped transects within roi
        identifier_column (str): column name of transect id column by deafult 'id' 
    Returns:
        dict: dict of np.arrays of transect start and end points
        of form {
            '<transect_id>': array([[start point],
                        [end point]]),}
    """
    features = []
    if identifier_column not in feature.columns:
        raise ValueError(f"{identifier_column} not in feature columns")
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(ignore_index=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: {str(row[identifier_column]): np.array(np.array(row.geometry.coords).tolist())},
        axis=1,
    )
    features = list(lat_lng)
    new_dict = {}
    for item in list(features):
        new_dict = {**new_dict, **item}
    return new_dict

def combine_geojson_files_with_dates(file_paths):
    """
    Combines multiple GeoJSON files into a single GeoDataFrame and assigns a date column 
    based on the filenames.

    Parameters:
    - file_paths (list of str): List of file paths to GeoJSON files.

    Returns:
    - GeoDataFrame: Combined GeoDataFrame containing all features from the input files, 
                    with an added date column.
    """
    gdfs = []
    for file in file_paths:
        gdf = gpd.read_file(file)
        if "z" in gdf.columns:
                gdf.drop(columns="z", inplace=True)
        date_str = os.path.basename(file).split("_")[0]
        gdf["date"] = datetime.strptime(date_str, "%Y%m%d")
        gdfs.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    return combined_gdf

# 0. Enter the locations of the shoreline, transects and the folder to save the raw timeseries csv to
shoreline_file = r"C:\Users\sf230\Downloads\no_filter_points.geojson"
transect_file = r"C:\Users\sf230\Downloads\transects_mark.geojson"
save_folder = r"C:\Users\sf230\Downloads" # this is where the csv will be saved
filename ="raw_transect_time_series.csv"

# 1. Read in the shoreline file
# shoreline_file = r"C:\Users\sf230\Downloads\Honolulu_shorelines4326\2023-06-01.shp"

shoreline_gdf = gpd.read_file(shoreline_file)
print(shoreline_gdf.head(2))
print(f"shoreline.crs: {shoreline_gdf.crs}")
shorelines_dict = {}

# 2. read in the transects file
transects_gdf = gpd.read_file(transect_file)
print(transects_gdf.head(2))
print(f"transects_gdf.crs: {transects_gdf.crs}")

# 3. Match the CRS of the transects to the shorelines (Note: crs cannot be epsg:4326 or 4327 )

# 3a) estimate the crs of the transects
crs = transects_gdf.estimate_utm_crs()
transects_gdf.to_crs(crs, inplace=True)
print(f"After using estimate utm crs transects crs: {transects_gdf.crs}" )
# 3b) match the crs of the shoreline to the transects
if shoreline_gdf.crs == crs:
    print("CRS match no need to convert shoreline crs")
else:
    shoreline_gdf.to_crs(crs, inplace=True)
print(f"After using estimate utm crs shoreline.crs: {shoreline_gdf.crs}")

print(f"Transects CRS {transects_gdf.crs} Shoreline CRS {shoreline_gdf.crs}")
print(f"transects gdf: {transects_gdf.head(2)}")


# 4. convert the transects to a dictionary {transect_name: [start_point, end_point], ...}
transects_dict = get_transect_points_dict(transects_gdf,identifier_column="OBJECTID")
# print(f"transects_dict {transects_dict}")
print(f"There are {len(transects_dict)} transects in the dictionary")


shorelines_dict = {}


# 5. convert the shorelines to a dictionary {date: [shoreline], ...}
shorelines_dict = convert_shoreline_gdf_to_dict(shoreline_gdf,date_format='%Y-%m-%dT%H:%M:%S%z')
print(f"Number of shorelines: {len(shorelines_dict['shorelines'])}")
print(f"Number of dates: {len(shorelines_dict['dates'])}")
print(f"Number of shorelines before conversion: {len(shoreline_gdf)}")
# print(f"First shoreline: {shorelines_dict['shorelines'][0]}")

#6. compute the intersection between each shoreline and each transect
cross_dist = compute_intersection_QC(shorelines_dict, transects_dict)
# this is a dictionary with the transect names as keys and the location along the transect where the shoreline intersected as the values

# 7. Save the raw timeseries to a CSV file\
save_raw_timesseries(shorelines_dict,cross_dist, save_folder,filename =filename, verbose=True)

