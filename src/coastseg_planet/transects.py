import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union
from typing import Tuple, Optional
import tqdm
import json
import datetime
import numpy as np
from shapely.geometry import MultiLineString
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon
import os
import shapely
from json import JSONEncoder
import pandas as pd

# Right now the function expects the extracted shorelines to be in this format
        # output[satname] = {
        #     "dates": output_timestamp,
        #     "shorelines": output_shoreline,
        #     "filename": output_filename,
        #     "cloud_cover": output_cloudcover,
        #     "geoaccuracy": output_geoaccuracy,
        #     "idx": output_idxkeep,
        #     "MNDWI_threshold": output_t_mndwi,
        # }
#
# These are then merged by output = SDS_tools.merge_output(output)

def intersect_transects(transects_path:str,shorelines_dict:dict,output_epsg:int,save_location:str):
    """
    Intersects transects with shorelines and saves the raw timeseries.

    Parameters:
    transects_path (str): The file path to the transects shapefile.
    save_location (str): The directory where the raw timeseries will be saved.

    Returns:
    None
    """
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf.to_crs(output_epsg, inplace=True)
    # if no id column in transetcs_gdf then create one
    if "id" not in transects_gdf.columns:
        transects_gdf["id"] = transects_gdf.index
        transects_gdf["id"] = transects_gdf["id"].astype(str)
    # turn the transects into a dictionary so that we can use it in the compute_intersection_QC function
    transects_dict = get_transect_points_dict(transects_gdf)
    # run compute_intersection_QC
    intersections_dict = compute_intersection_QC(shorelines_dict, transects_dict)
    # save the raw timeseries
    save_raw_timesseries(shorelines_dict,intersections_dict, save_location,verbose=True)

def convert_transect_ids_to_rows(df):
    """
    Reshapes the timeseries data so that transect IDs become rows.

    Args:
    - df (DataFrame): Input data with transect IDs as columns.

    Returns:
    - DataFrame: Reshaped data with transect IDs as rows.
    """
    reshaped_df = df.melt(
        id_vars="dates", var_name="transect_id", value_name="cross_distance"
    )
    return reshaped_df.dropna()
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


def get_raw_timeseries(extracted_shorelines:dict, cross_distance_transects:dict):
    cross_distance_df = get_cross_distance_df(
        extracted_shorelines, cross_distance_transects
    )
    cross_distance_df.dropna(axis="columns", how="all", inplace=True)
    # this converts it to format dates, transect_id, cross_distance as the columns
    # timeseries_df = convert_transect_ids_to_rows(cross_distance_df)
    # timeseries_df = timeseries_df.sort_values('dates')
    return cross_distance_df


def save_raw_timesseries(shorelines_dict,intersections_dict, save_location,verbose=True):
    raw_timeseries= get_raw_timeseries(shorelines_dict,intersections_dict)
    filepath = os.path.join(save_location, "raw_transect_time_series.csv")
    raw_timeseries.to_csv(filepath, sep=",",index=False)
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
    # df = pd.DataFrame(transects_csv)
    # this would add the satellite the image was captured on to the timeseries
    # df['satname'] = extracted_shorelines["satname"]
    return pd.DataFrame(transects_csv)


def convert_transect_ids_to_rows(df):
    """
    Reshapes the timeseries data so that transect IDs become rows.

    Args:
    - df (DataFrame): Input data with transect IDs as columns.

    Returns:
    - DataFrame: Reshaped data with transect IDs as rows.
    """
    reshaped_df = df.melt(
        id_vars="dates", var_name="transect_id", value_name="cross_distance"
    )
    return reshaped_df.dropna()

def get_seaward_points_gdf(transects_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame containing the seaward points from a given GeoDataFrame containing transects.
    CRS will always be 4326.

    Parameters:
    - transects_gdf: A GeoDataFrame containing transect data.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the seaward points for all of the transects.
    Contains columns transect_id and geometry in crs 4326
    """
    # Set transects crs to epsg:4326 if it is not already. Tide model requires crs 4326
    if transects_gdf.crs is None:
        transects_gdf = transects_gdf.set_crs("epsg:4326")
    else:
        transects_gdf = transects_gdf.to_crs("epsg:4326")

    # Prepare data for the new GeoDataFrame
    data = []
    for index, row in transects_gdf.iterrows():
        points = list(row["geometry"].coords)
        seaward_point = Point(points[1]) if len(points) > 1 else Point()

        # Append data for each transect to the data list
        data.append({"transect_id": row["id"], "geometry": seaward_point})

    # Create the new GeoDataFrame
    seaward_points_gdf = gpd.GeoDataFrame(data, crs="epsg:4326")

    return seaward_points_gdf

def merge_dataframes(df1, df2, columns_to_merge_on=set(["transect_id", "dates"])):
    """
    Merges two DataFrames based on column names provided in columns_to_merge_on by default
    merges on "transect_id", "dates".

    Args:
    - df1 (DataFrame): First DataFrame.
    - df2 (DataFrame): Second DataFrame.
    - columns_to_merge_on(collection): column names to merge on
    Returns:
    - DataFrame: Merged data.
    """
    merged_df = pd.merge(df1, df2, on=list(columns_to_merge_on), how="inner")
    return merged_df.drop_duplicates(ignore_index=True)

def intersect_with_buffered_transects(points_gdf, transects, buffer_distance = 0.00000001):
    """
    Intersects points from a GeoDataFrame with another GeoDataFrame and exports the result to a new GeoDataFrame, retaining all original attributes.
    Additionally, returns the points that do not intersect with the buffered transects.
    
    Parameters:
    - points_gdf: GeoDataFrame - The input GeoDataFrame containing the points to be intersected.
    - transects: GeoDataFrame - The GeoDataFrame representing the transects to intersect with.
    - buffer_distance: float - The buffer distance to apply to the transects (default: 0.00000001).
    
    Returns:
    - filtered: GeoDataFrame - The resulting GeoDataFrame containing the intersected points within the buffered transects.
    - dropped_rows: GeoDataFrame - The rows that were filtered out during the intersection process.
    """
    
    buffered_lines_gdf = transects.copy()  # Create a copy to preserve the original data
    buffered_lines_gdf['geometry'] = transects.geometry.buffer(buffer_distance)
    points_within_buffer = points_gdf[points_gdf.geometry.within(buffered_lines_gdf.unary_union)]
    
    grouped = points_within_buffer.groupby('transect_id')
    
    # Filter out points not within their respective buffered transect
    filtered = grouped.filter(lambda x: x.geometry.within(buffered_lines_gdf[buffered_lines_gdf['id'].isin(x['transect_id'])].unary_union).all())

    # Identify the dropped rows by comparing the original dataframe within the buffer and the filtered results
    dropped_rows = points_gdf[~points_gdf.index.isin(filtered.index)]

    return filtered, dropped_rows

def filter_points_outside_transects(merged_timeseries_gdf:gpd.GeoDataFrame, transects_gdf:gpd.GeoDataFrame, save_location: str, name: str = ""):
    """
    Filters points outside of transects from a merged timeseries GeoDataFrame.

    Args:
        merged_timeseries_gdf (GeoDataFrame): The merged timeseries GeoDataFrame containing the shore x and shore y columns that indicated where the shoreline point was along the transect
        transects_gdf (GeoDataFrame): The transects GeoDataFrame used for filtering.
        save_location (str): The directory where the filtered points will be saved.
        name (str, optional): The name to be appended to the saved file. Defaults to "".

    Returns:
        tuple: A tuple containing the filtered merged timeseries GeoDataFrame and a DataFrame of dropped points.

    """
    extension = "" if name == "" else f'{name}_'
    timeseries_df = pd.DataFrame(merged_timeseries_gdf)
    timeseries_df.drop(columns=['geometry'], inplace=True)
    # estimate crs of transects
    utm_crs = merged_timeseries_gdf.estimate_utm_crs()
    # intersect the points with the transects
    filtered_merged_timeseries_gdf_utm, dropped_points_df = intersect_with_buffered_transects(
        merged_timeseries_gdf.to_crs(utm_crs), transects_gdf.to_crs(utm_crs))
    # Get a dataframe containing the points that were filtered out from the time series because they were not on the transects
    dropped_points_df.drop(columns=['geometry']).to_csv(os.path.join(save_location, f"{extension}dropped_points_time_series.csv"), index=False)
    # convert back to same crs as original merged_timeseries_gdf
    merged_timeseries_gdf = filtered_merged_timeseries_gdf_utm.to_crs(merged_timeseries_gdf.crs)
    return merged_timeseries_gdf, dropped_points_df

def add_shore_points_to_timeseries(timeseries_data: pd.DataFrame,
                                 transects: gpd.GeoDataFrame,)->pd.DataFrame:
    """
    Edits the transect_timeseries_merged.csv or transect_timeseries_tidally_corrected.csv
    so that there are additional columns with lat (shore_y) and lon (shore_x).
    
    
    inputs:
    timeseries_data (pd.DataFrame): dataframe containing the data from transect_timeseries_merged.csv
    transects (gpd.GeoDataFrame): geodataframe containing the transects 
    
    returns:
    pd.DataFrame: the new timeseries_data with the lat and lon columns
    """
    
    ##Gonna do this in UTM to keep the math simple...problems when we get to longer distances (10s of km)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)
    
    ##need some placeholders
    shore_x_vals = [None]*len(timeseries_data)
    shore_y_vals = [None]*len(timeseries_data)
    timeseries_data['shore_x'] = shore_x_vals
    timeseries_data['shore_y'] = shore_y_vals

    ##loop over all transects
    for i in range(len(transects_utm)):
        transect = transects_utm.iloc[i]
        transect_id = transect['id']
        first = transect.geometry.coords[0]
        last = transect.geometry.coords[1]
        
        idx = timeseries_data['transect_id'].str.contains(transect_id)
        ##in case there is a transect in the config_gdf that doesn't have any intersections
        ##skip that transect
        if np.any(idx):
            timeseries_data_filter = timeseries_data[idx]
        else:
            continue

        idxes = timeseries_data_filter.index
        distances = timeseries_data_filter['cross_distance']

        angle = np.arctan2(last[1] - first[1], last[0] - first[0])

        shore_x_utm = first[0]+distances*np.cos(angle)
        shore_y_utm = first[1]+distances*np.sin(angle)
        points_utm = [shapely.Point(xy) for xy in zip(shore_x_utm, shore_y_utm)]

        #conversion from utm to wgs84, put them in the transect_timeseries csv and utm gdf
        dummy_gdf_utm = gpd.GeoDataFrame({'geometry':points_utm},
                                         crs=utm_crs)
        dummy_gdf_wgs84 = dummy_gdf_utm.to_crs(org_crs)

        points_wgs84 = [shapely.get_coordinates(p) for p in dummy_gdf_wgs84.geometry]
        points_wgs84 = np.array(points_wgs84)
        points_wgs84 = points_wgs84.reshape(len(points_wgs84),2)
        x_wgs84 = points_wgs84[:,0]
        y_wgs84 = points_wgs84[:,1]
        timeseries_data.loc[idxes,'shore_x'] = x_wgs84
        timeseries_data.loc[idxes,'shore_y'] = y_wgs84

    return timeseries_data

def filter_dropped_points_out_of_timeseries(timeseries_df:pd.DataFrame, dropped_points_df:pd.DataFrame)->pd.DataFrame:
    """
    Filter out dropped points from a timeseries dataframe.

    Args:
        timeseries_df (pandas.DataFrame): The timeseries dataframe to filter.
        dropped_points_df (pandas.DataFrame): The dataframe containing dropped points information.

    Returns:
        pandas.DataFrame: The filtered timeseries dataframe with dropped points set to NaN.
    """
    # Iterate through unique transect ids from drop_df to avoid setting the same column multiple times
    for t_id in dropped_points_df['transect_id'].unique():
        # Find all the dates associated with this transect_id in dropped_points_df
        dates_to_drop = dropped_points_df.loc[dropped_points_df['transect_id'] == t_id, 'dates']
        timeseries_df.loc[timeseries_df['dates'].isin(dates_to_drop), t_id] = np.nan
    return timeseries_df

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

def create_complete_line_string(points):
    """
    Create a complete LineString from a list of points.

    Args:
        points (numpy.ndarray): An array of points representing the coordinates.

    Returns:
        LineString: A LineString object representing the complete line.

    Raises:
        None.

    """
    # Ensure all points are unique to avoid redundant looping
    unique_points = np.unique(points, axis=0)
    
    # Start with the first point in the list
    if len(unique_points) == 0:
        return None  # Return None if there are no points
    
    starting_point = unique_points[0]
    current_point = starting_point
    sorted_points = [starting_point]
    visited_points = {tuple(starting_point)}

    # Repeat until all points are visited
    while len(visited_points) < len(unique_points):
        nearest_distance = np.inf
        nearest_point = None
        for point in unique_points:
            if tuple(point) in visited_points:
                continue  # Skip already visited points
            # Calculate the distance to the current point
            distance = np.linalg.norm(point - current_point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_point = point

        # Break if no unvisited nearest point was found (should not happen if all points are unique)
        if nearest_point is None:
            break

        sorted_points.append(nearest_point)
        visited_points.add(tuple(nearest_point))
        current_point = nearest_point

    # Convert the sorted list of points to a LineString
    return LineString(sorted_points)

def order_linestrings_gdf(gdf,dates, output_crs='epsg:4326'):
    """
    Orders the linestrings in a GeoDataFrame by creating complete line strings from the given points.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing linestrings.
        dates (list): The list of dates corresponding to the linestrings.
        output_crs (str): The output coordinate reference system (CRS) for the GeoDataFrame. Default is 'epsg:4326'.

    Returns:
        GeoDataFrame: The ordered GeoDataFrame with linestrings.

    """
    gdf = gdf.copy()
    # Convert to the output CRS
    if gdf.crs is not None:
        gdf.to_crs(output_crs, inplace=True)
    else:
        gdf.set_crs(output_crs, inplace=True)
        
    all_points = [shapely.get_coordinates(p) for p in gdf.geometry]
    lines = []
    for points in all_points:
        line_string = create_complete_line_string(points)
        lines.append(line_string)
    
    gdf = gpd.GeoDataFrame({'geometry': lines,'date': dates},crs=output_crs)
    return gdf

def convert_points_to_linestrings(gdf, group_col='date', output_crs='epsg:4326') -> gpd.GeoDataFrame:
    """
    Convert points to LineStrings.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame containing points.
        group_col (str): The column to group the GeoDataFrame by (default is 'date').
        output_crs (str): The coordinate reference system for the output GeoDataFrame (default is 'epsg:4326').

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame containing LineStrings created from the points.
    """
    # Group the GeoDataFrame by date
    gdf = gdf.copy()
    # Convert to the output CRS
    if gdf.crs is not None:
        gdf.to_crs(output_crs, inplace=True)
    else:
        gdf.set_crs(output_crs, inplace=True)
    grouped = gdf.groupby(group_col)
    # For each group, ensure there are at least two points so that a LineString can be created
    filtered_groups = grouped.filter(lambda g: g[group_col].count() > 1)
    if len(filtered_groups) <= 0:
        print("No groups contain at least two points, so no LineStrings can be created.")
        return gpd.GeoDataFrame(columns=['geometry'])
    # Recreate the groups as a geodataframe
    grouped_gdf = gpd.GeoDataFrame(filtered_groups, geometry='geometry')
    linestrings = grouped_gdf.groupby(group_col).apply(lambda g: LineString(g.geometry.tolist()))
    # Create a new GeoDataFrame from the LineStrings
    linestrings_gdf = gpd.GeoDataFrame(linestrings, columns=['geometry'],)
    linestrings_gdf.reset_index(inplace=True)
    
    # order the linestrings so that they are continuous
    linestrings_gdf = order_linestrings_gdf(linestrings_gdf,linestrings_gdf['date'],output_crs=output_crs)
    
    return linestrings_gdf

def convert_date_gdf(gdf):
    """
    Convert date columns in a GeoDataFrame to datetime format.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame.

    Returns:
        GeoDataFrame: The converted GeoDataFrame with date columns in datetime format.
    """
    gdf = gdf.copy()
    if 'dates' in gdf.columns:
        gdf['dates'] = pd.to_datetime(gdf['dates']).dt.tz_localize('UTC')
        gdf['dates'] = pd.to_datetime(gdf['dates']).dt.tz_convert(None)
    if 'date' in gdf.columns:
        gdf['date'] = pd.to_datetime(gdf['date']).dt.tz_localize('UTC')
        gdf['date'] = pd.to_datetime(gdf['date']).dt.tz_convert(None)
    gdf = stringify_datetime_columns(gdf)
    return gdf

def add_lat_lon_to_timeseries(merged_timeseries_df, transects_gdf,timeseries_df,
                              save_location:str,
                              only_keep_points_on_transects:bool=False,
                              extension:str=""):
    """
    Adds latitude and longitude coordinates to a timeseries dataframe based on shoreline positions.

    Args:
        merged_timeseries_df (pandas.DataFrame): The timeseries dataframe to add latitude and longitude coordinates to.
        transects_gdf (geopandas.GeoDataFrame): The geodataframe containing transect information.
        timeseries_df (pandas.DataFrame): The original timeseries dataframe.This is a matrix of dates x transect id with the cross shore distance as the values.
        save_location (str): The directory path to save the output files.
        only_keep_points_on_transects (bool, optional): Whether to keep only the points that fall on the transects. 
                                                  Defaults to False.
        extension (str, optional): An extension to add to the output filenames. Defaults to "".
        

    Returns:
        pandas.DataFrame: The updated timeseries dataframe with latitude and longitude coordinates.

    """
    ext = "" if extension=="" else f'{extension}'
    
    # add the shoreline position as an x and y coordinate to the csv called shore_x and shore_y
    merged_timeseries_df = add_shore_points_to_timeseries(merged_timeseries_df, transects_gdf)
    
    # convert to geodataframe
    merged_timeseries_gdf = gpd.GeoDataFrame(
        merged_timeseries_df, 
        geometry=[Point(xy) for xy in zip(merged_timeseries_df['shore_x'], merged_timeseries_df['shore_y'])], 
        crs="EPSG:4326"
    )
    merged_timeseries_gdf.to_crs("EPSG:4326",inplace=True)
    if only_keep_points_on_transects:
        merged_timeseries_gdf,dropped_points_df = filter_points_outside_transects(merged_timeseries_gdf,transects_gdf,save_location,ext)
        if not dropped_points_df.empty:
            timeseries_df = filter_dropped_points_out_of_timeseries(timeseries_df, dropped_points_df)
    
    # save the time series of along shore points as points to a geojson (saves shore_x and shore_y as x and y coordinates in the geojson)
    cross_shore_pts = convert_date_gdf(merged_timeseries_gdf.drop(columns=['x','y','shore_x','shore_y','cross_distance']).to_crs('epsg:4326'))
    # rename the dates column to date
    cross_shore_pts.rename(columns={'dates':'date'},inplace=True)
    
    # Create 2D vector of shorelines from where each shoreline intersected the transect
    if cross_shore_pts.empty:
        print("No points were found on the transects. Skipping the creation of the transect_time_series_points.geojson and transect_time_series_vectors.geojson files")
        return merged_timeseries_df,timeseries_df
    
    new_gdf_shorelines_wgs84=convert_points_to_linestrings(cross_shore_pts, group_col='date', output_crs='epsg:4326')
    new_gdf_shorelines_wgs84_path = os.path.join(save_location, f'{ext}_transect_time_series_vectors.geojson')
    new_gdf_shorelines_wgs84.to_file(new_gdf_shorelines_wgs84_path)
    
    # save the merged time series that includes the shore_x and shore_y columns to a geojson file and a  csv file
    merged_timeseries_gdf_cleaned = convert_date_gdf(merged_timeseries_gdf.drop(columns=['x','y','shore_x','shore_y','cross_distance']).rename(columns={'dates':'date'}).to_crs('epsg:4326'))
    merged_timeseries_gdf_cleaned.to_file(os.path.join(save_location, f"{ext}_transect_time_series_points.geojson"), driver='GeoJSON')
    merged_timeseries_df = pd.DataFrame(merged_timeseries_gdf.drop(columns=['geometry']))

    return merged_timeseries_df,timeseries_df

def get_transect_settings(settings: dict) -> dict:
    transect_settings = {}
    transect_settings["max_std"] = settings.get("max_std")
    transect_settings["min_points"] = settings.get("min_points")
    transect_settings["along_dist"] = settings.get("along_dist")
    transect_settings["max_range"] = settings.get("max_range")
    transect_settings["min_chainage"] = settings.get("min_chainage")
    transect_settings["multiple_inter"] = settings.get("multiple_inter")
    transect_settings["prc_multiple"] = settings.get("prc_multiple")
    return transect_settings


def to_file(data: dict, filepath: str) -> None:
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

def save_transects(
    save_location: str,
    cross_distance_transects: dict,
    extracted_shorelines: dict,
    settings: dict,
    transects_gdf:gpd.GeoDataFrame,
    drop_intersection_pts = False,
) -> None:
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        roi_id (str): The ID of the ROI.
        save_location (str): The directory path to save the transect data.
        cross_distance_transects (dict): Dictionary containing cross distance transects data.
        extracted_shorelines (dict): Dictionary containing extracted shorelines data.
        drop_intersection_pts (bool): If True, keep only the shoreline points that are on the transects. Default is False.
        - This will generated a file called "dropped_points_time_series.csv" that contains the points that were filtered out. If only_keep_points_on_transects is True.
        - Any shoreline points that were not on the transects will be removed from "raw_transect_time_series.csv" by setting those values to NaN.v If only_keep_points_on_transects is True.
        - The "raw_transect_time_series_merged.csv" will not contain any points that were not on the transects. If only_keep_points_on_transects is True.

    Returns:
        None.
    """    
    cross_distance_df =get_cross_distance_df(
        extracted_shorelines, cross_distance_transects
    )
    cross_distance_df.dropna(axis="columns", how="all", inplace=True)
    
    # get the last point (aka the seaward point) from each transect
    seaward_points = get_seaward_points_gdf(transects_gdf)
    timeseries_df = convert_transect_ids_to_rows(cross_distance_df)
    timeseries_df = timeseries_df.sort_values('dates')
    print(f"timeseries.columns: {timeseries_df.columns}")
    print(f"seaward_points.columns: {seaward_points.columns}")
    merged_timeseries_df=merge_dataframes(timeseries_df, seaward_points,columns_to_merge_on=["transect_id"])
    merged_timeseries_df['x'] = merged_timeseries_df['geometry'].apply(lambda geom: geom.x)
    merged_timeseries_df['y'] = merged_timeseries_df['geometry'].apply(lambda geom: geom.y)
    merged_timeseries_df.drop('geometry', axis=1, inplace=True)

    # re-order columns
    merged_timeseries_df = merged_timeseries_df[['dates', 'x', 'y', 'transect_id', 'cross_distance']]
    # add the shore_x and shore_y columns to the merged time series which are the x and y coordinates of the shore points along the transects
    merged_timeseries_df,timeseries_df = add_lat_lon_to_timeseries(merged_timeseries_df, transects_gdf.to_crs('epsg:4326'),cross_distance_df,
                              save_location,
                              drop_intersection_pts,
                              "raw")
    # save the raw transect time series which contains the columns ['dates', 'x', 'y', 'transect_id', 'cross_distance','shore_x','shore_y']  to file
    filepath = os.path.join(save_location, "raw_transect_time_series_merged.csv")
    merged_timeseries_df.to_csv(filepath, sep=",",index=False) 
    
    filepath = os.path.join(save_location, "raw_transect_time_series.csv")
    timeseries_df.to_csv(filepath, sep=",",index=False)
    # save transect settings to file
    transect_settings = get_transect_settings(settings)
    transect_settings_path = os.path.join(save_location, "transects_settings.json")
    to_file(transect_settings, transect_settings_path)
    save_path = os.path.join(save_location, "transects_cross_distances.json")
    to_file(cross_distance_transects, save_path)

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
            if isinstance(geometry, MultiLineString):
                for line in geometry.geoms:
                    shorelines_array = np.array(line.coords)
                    shorelines.append(shorelines_array)
                    dates.append(date_str)
            else:
                shorelines_array = np.array(geometry.coords)
                shorelines.append(shorelines_array)
                dates.append(date_str)

    shorelines_dict = {'dates': dates, 'shorelines': shorelines}
    return shorelines_dict

# this is an optimized version of the function this 65% faster than the original
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

    cross_dist = {}

    shorelines = output["shorelines"]
    transect_keys = list(transects.keys())
    if use_progress_bar:
        transect_keys = tqdm.tqdm(transect_keys, desc="Computing transect shoreline intersections")

    for key in transect_keys:
        std_intersect = np.full(len(shorelines), np.nan)
        med_intersect = np.full(len(shorelines), np.nan)
        max_intersect = np.full(len(shorelines), np.nan)
        min_intersect = np.full(len(shorelines), np.nan)
        n_intersect = np.full(len(shorelines), np.nan)

        transect_start = transects[key][0, :]
        transect_end = transects[key][-1, :]
        transect_vector = transect_end - transect_start
        transect_length = np.linalg.norm(transect_vector)
        transect_unit_vector = transect_vector / transect_length
        rotation_matrix = np.array([[transect_unit_vector[0], transect_unit_vector[1]],
                                    [-transect_unit_vector[1], transect_unit_vector[0]]])

        for i, shoreline in enumerate(shorelines):
            if len(shoreline) == 0:
                continue

            shoreline_shifted = shoreline - transect_start
            shoreline_rotated = np.dot(rotation_matrix, shoreline_shifted.T).T

            d_line = np.abs(shoreline_rotated[:, 1])
            d_origin = np.linalg.norm(shoreline_shifted, axis=1)
            idx_close = (d_line <= along_dist) & (d_origin <= 1000)

            if not np.any(idx_close):
                continue
            
            valid_points = shoreline_rotated[idx_close, 0]
            valid_points = valid_points[valid_points >= min_chainage]

            if np.sum(~np.isnan(valid_points)) < min_points:
                continue
                        
            std_intersect[i] = np.nanstd(valid_points)
            med_intersect[i] = np.nanmedian(valid_points)
            max_intersect[i] = np.nanmax(valid_points)
            min_intersect[i] = np.nanmin(valid_points)
            n_intersect[i] = np.sum(~np.isnan(valid_points))

        condition1 = std_intersect <= max_std
        condition2 = (max_intersect - min_intersect) <= max_range
        condition3 = n_intersect >= min_points
        idx_good = condition1 & condition2 & condition3

        if multiple_inter == "auto":
            prc_over = np.sum(std_intersect > max_std) / len(std_intersect)
            if prc_over > prc_multiple:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            else:
                med_intersect[~idx_good] = np.nan
        elif multiple_inter == "max":
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan
        elif multiple_inter == "nan":
            med_intersect[~idx_good] = np.nan
        else:
            raise ValueError("The multiple_inter parameter can only be: nan, max, or auto.")

        cross_dist[key] = med_intersect

    return cross_dist



# def compute_intersection_QC_original(output, 
#                             transects,
#                             along_dist = 25, 
#                             min_points = 3,
#                             max_std = 15,
#                             max_range = 30, 
#                             min_chainage = -100, 
#                             multiple_inter ="auto",
#                             prc_multiple=None, 
#                             use_progress_bar: bool = True):
#     """
    
#     More advanced function to compute the intersection between the 2D mapped shorelines
#     and the transects. Produces more quality-controlled time-series of shoreline change.

#     Arguments:
#     -----------
#         output: dict
#             contains the extracted shorelines and corresponding dates.
#         transects: dict
#             contains the X and Y coordinates of the transects (first and last point needed for each
#             transect).
#         along_dist: int (in metres)
#             alongshore distance to calculate the intersection (median of points
#             within this distance).
#         min_points: int
#             minimum number of shoreline points to calculate an intersection.
#         max_std: int (in metres)
#             maximum std for the shoreline points when calculating the median,
#             if above this value then NaN is returned for the intersection.
#         max_range: int (in metres)
#             maximum range for the shoreline points when calculating the median,
#             if above this value then NaN is returned for the intersection.
#         min_chainage: int (in metres)
#             furthest landward of the transect origin that an intersection is
#             accepted, beyond this point a NaN is returned.
#         multiple_inter: mode for removing outliers ('auto', 'nan', 'max').
#         prc_multiple: float, optional
#             percentage to use in 'auto' mode to switch from 'nan' to 'max'.
#         use_progress_bar(bool,optional). Defaults to True. If true uses tqdm to display the progress for iterating through transects.
#             False, means no progress bar is displayed.

#     Returns:
#     -----------
#         cross_dist: dict
#             time-series of cross-shore distance along each of the transects. These are not tidally
#             corrected.
#     """

#     # initialise dictionary with intersections for each transect
#     cross_dist = dict([])

#     shorelines = output["shorelines"]

#     # loop through each transect
#     transect_keys = transects.keys()
#     if use_progress_bar:
#         transect_keys = tqdm(
#             transect_keys, desc="Computing transect shoreline intersections"
#         )

#     for key in transect_keys:
#         # initialise variables
#         std_intersect = np.zeros(len(shorelines))
#         med_intersect = np.zeros(len(shorelines))
#         max_intersect = np.zeros(len(shorelines))
#         min_intersect = np.zeros(len(shorelines))
#         n_intersect = np.zeros(len(shorelines))

#         # loop through each shoreline
#         for i in range(len(shorelines)):
#             sl = shorelines[i]

#             # in case there are no shoreline points
#             if len(sl) == 0:
#                 std_intersect[i] = np.nan
#                 med_intersect[i] = np.nan
#                 max_intersect[i] = np.nan
#                 min_intersect[i] = np.nan
#                 n_intersect[i] = np.nan
#                 continue

#             # compute rotation matrix
#             X0 = transects[key][0, 0]
#             Y0 = transects[key][0, 1]
#             temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
#             phi = np.arctan2(temp[1], temp[0])
#             Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

#             # calculate point to line distance between shoreline points and the transect
#             p1 = np.array([X0, Y0])
#             p2 = transects[key][-1, :]
#             d_line = np.abs(np.cross(p2 - p1, sl - p1) / np.linalg.norm(p2 - p1))
#             # calculate the distance between shoreline points and the origin of the transect
#             d_origin = np.array([np.linalg.norm(sl[k, :] - p1) for k in range(len(sl))])
#             # find the shoreline points that are close to the transects and to the origin
#             idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
#             idx_close = np.where(idx_dist)[0]

#             # in case there are no shoreline points close to the transect
#             if len(idx_close) == 0:
#                 std_intersect[i] = np.nan
#                 med_intersect[i] = np.nan
#                 max_intersect[i] = np.nan
#                 min_intersect[i] = np.nan
#                 n_intersect[i] = np.nan
#             else:
#                 # change of base to shore-normal coordinate system
#                 xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
#                     np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
#                 )
#                 xy_rot = np.matmul(Mrot, xy_close)
#                 # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
#                 xy_rot[0, xy_rot[0, :] < min_chainage] = np.nan

#                 # compute std, median, max, min of the intersections
#                 if not np.all(np.isnan(xy_rot[0, :])):
#                     std_intersect[i] = np.nanstd(xy_rot[0, :])
#                     med_intersect[i] = np.nanmedian(xy_rot[0, :])
#                     max_intersect[i] = np.nanmax(xy_rot[0, :])
#                     min_intersect[i] = np.nanmin(xy_rot[0, :])
#                     n_intersect[i] = len(xy_rot[0, :])
#                 else:
#                     std_intersect[i] = np.nan
#                     med_intersect[i] = np.nan
#                     max_intersect[i] = np.nan
#                     min_intersect[i] = np.nan
#                     n_intersect[i] = 0

#         # quality control the intersections using dispersion metrics (std and range)
#         condition1 = std_intersect <= max_std
#         condition2 = (max_intersect - min_intersect) <= max_range
#         condition3 = n_intersect >= min_points
#         idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)

#         # decide what to do with the intersections with high dispersion
#         if multiple_inter == "auto":
#             # compute the percentage of data points where the std is larger than the user-defined max
#             prc_over = np.sum(std_intersect > max_std) / len(std_intersect)
#             # if more than a certain percentage is above, use the maximum intersection
#             if prc_over > prc_multiple:
#                 med_intersect[~idx_good] = max_intersect[~idx_good]
#                 med_intersect[~condition3] = np.nan
#             # otherwise put a nan
#             else:
#                 med_intersect[~idx_good] = np.nan

#         elif multiple_inter == "max":
#             med_intersect[~idx_good] = max_intersect[~idx_good]
#             med_intersect[~condition3] = np.nan

#         elif multiple_inter == "nan":
#             med_intersect[~idx_good] = np.nan

#         else:
#             raise Exception(
#                 "The multiple_inter parameter can only be: nan, max, or auto."
#             )

#         # store in dict
#         cross_dist[key] = med_intersect

#     return cross_dist

def get_transect_points_dict(feature: gpd.GeoDataFrame) -> dict:
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
    Returns:
        dict: dict of np.arrays of transect start and end points
        of form {
            '<transect_id>': array([[start point],
                        [end point]]),}
    """
    features = []
    # Use explode to break multilinestrings in linestrings
    feature_exploded = feature.explode(ignore_index=True)
    # For each linestring portion of feature convert to lat,lon tuples
    lat_lng = feature_exploded.apply(
        lambda row: {str(row.id): np.array(np.array(row.geometry.coords).tolist())},
        axis=1,
    )
    features = list(lat_lng)
    new_dict = {}
    for item in list(features):
        new_dict = {**new_dict, **item}
    return new_dict

def compute_transects_from_roi(
    extracted_shorelines: dict,
    transects_gdf: gpd.GeoDataFrame,
    settings: dict,
) -> dict:
    """Computes the intersection between the 2D shorelines and the shore-normal.
        transects. It returns time-series of cross-shore distance along each transect.
    Args:
        extracted_shorelines (dict): contains the extracted shorelines and corresponding metadata
        transects_gdf (gpd.GeoDataFrame): transects in ROI with crs = output_crs in settings
        settings (dict): settings dict with keys
                    'along_dist': int
                        alongshore distance considered calculate the intersection
    Returns:
        dict:  time-series of cross-shore distance along each of the transects.
               Not tidally corrected.
    """
    # create dict of numpy arrays of transect start and end points

    transects = common.get_transect_points_dict(transects_gdf)
    # cross_distance: along-shore distance over which to consider shoreline points to compute median intersection (robust to outliers)
    cross_distance = compute_intersection_QC(extracted_shorelines, transects, settings)
    return cross_distance



    def get_cross_distance(
        self,
        roi_id: str,
        transects_in_roi_gdf: gpd.GeoDataFrame,
        settings: dict,
        output_epsg: int,
    ) -> Tuple[float, Optional[str]]:
        """
        Compute the cross shore distance of transects and extracted shorelines for a given ROI.

        Parameters:
        -----------
        roi_id : str
            The ID of the ROI to compute the cross shore distance for.
        transects_in_roi_gdf : gpd.GeoDataFrame
            All the transects in the ROI. Must contain the columns ["id", "geometry"]
        settings : dict
            A dictionary of settings to be used in the computation.
        output_epsg : int
            The EPSG code of the output projection.

        Returns:
        --------
        Tuple[float, Optional[str]]
            The computed cross shore distance, or 0 if there was an issue in the computation.
            The reason for failure, or '' if the computation was successful.
        """
        failure_reason = ""
        cross_distance = 0
        
        transects_in_roi_gdf = transects_in_roi_gdf.loc[:, ["id", "geometry"]]
        
        if transects_in_roi_gdf.empty:
            failure_reason = f"No transects intersect for the ROI {roi_id}"
            return cross_distance, failure_reason

        # Get extracted shorelines object for the currently selected ROI
        roi_extracted_shoreline = self.rois.get_extracted_shoreline(roi_id)

        if roi_extracted_shoreline is None:
            failure_reason = f"No extracted shorelines were found for the ROI {roi_id}"
        else:
            # Convert transects_in_roi_gdf to output_crs from settings
            transects_in_roi_gdf = transects_in_roi_gdf.to_crs(output_epsg)
            # Compute cross shore distance of transects and extracted shorelines
            extracted_shorelines_dict = roi_extracted_shoreline.dictionary
            cross_distance = compute_transects_from_roi(
                extracted_shorelines_dict,
                transects_in_roi_gdf,
                settings,
            )
            if cross_distance == 0:
                failure_reason = "Cross distance computation failed"

        return cross_distance, failure_reason

def compute_transects_per_roi(roi_gdf: gpd.GeoDataFrame, transects_gdf: gpd.GeoDataFrame, settings: dict, roi_id: str, output_epsg: int) -> None:
    """
    Computes the cross distance for transects within a specific region of interest (ROI).
    Args:
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame containing the ROIs.
        transects_gdf (gpd.GeoDataFrame): GeoDataFrame containing the transects.
        settings (dict): Dictionary of settings.
        roi_id (str): ID of the ROI.
        output_epsg (int): EPSG code for the output coordinate reference system.
    Returns:
        None: The cross distance is computed and logged. If the cross distance is 0, a warning message is logged.
    """
    # save cross distances by ROI id
    transects_in_roi_gdf = transects_gdf[
        transects_gdf.intersects(roi_gdf.unary_union)
    ]
    cross_distance, failure_reason = self.get_cross_distance(
        str(roi_id), transects_in_roi_gdf, settings, output_epsg
    )
    if cross_distance == 0:
        print(f"{failure_reason} for ROI {roi_id}")
    return cross_distance