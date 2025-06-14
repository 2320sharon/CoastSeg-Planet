import asyncio
import configparser
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely import geometry
from skimage.transform import resize
from tqdm.asyncio import tqdm_asyncio

import planet
from planet import (
    collect,
    data_filter,
)


async def query_planet_items(
    session: planet.Session,
    roi,
    start_date,
    end_date,
    item_type="PSScene",
    limit=10000,
    **kwargs,
):
    """
    Query the Planet Data API and return the list of items matching the criteria.

    Args:
        session (planet.Session): The Planet session object.
        roi: The region of interest (GeoJSON or geometry).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        item_type (str): The item type to search for (default: 'PSScene').
        limit (int): Maximum number of items to return (default: 10000).
        **kwargs: Additional filter arguments (e.g., cloud_cover).

    Returns:
        list: List of items returned by the API.
    """
    defaults = {
        "cloud_cover": 0.99,
    }
    defaults.update(kwargs)

    data_client = session.client("data")
    combined_filter = create_combined_filter(roi, start_date, end_date, **defaults)
    request = await data_client.create_search(
        name="temp_search", search_filter=combined_filter, item_types=[item_type]
    )
    items = data_client.run_search(search_id=request["id"], limit=limit)
    item_list = [i async for i in items]

    if not item_list:
        print(
            f"No items found that matched the search criteria.\n"
            f"  start_date: {start_date}\n"
            f"  end_date: {end_date}\n"
            f"  cloud_cover: {defaults.get('cloud_cover', 0.99)}"
        )
    return item_list


async def download_multiple_orders_in_parallel(order_list):
    """
    Takes a list of dictionaries representing orders and downloads them in parallel,
    with a progress bar showing the progress of all orders.

    Args:
        order_list (list of dicts): A list of dictionaries, where each dictionary is formatted like the output of `make_order_dict`.

    Returns:
        None
    """

    tasks = []

    for order in order_list:
        # Load ROI GeoDataFrame from file path provided in the order dictionary
        roi = gpd.read_file(order["roi_path"])

        print(f"Order received with name {order['order_name']}.")

        # If 'tools' is present, pass it to download_order_by_name, else use booleans
        task_kwargs = dict(
            order_name=order["order_name"],
            output_path=order["destination"],
            roi=roi,
            start_date=order["start_date"],
            end_date=order["end_date"],
            cloud_cover=order.get("cloud_cover", 0.7),  # Default to 0.7 if not provided
            min_area_percentage=order.get(
                "min_area_percentage", 0.7
            ),  # Default to 0.7 if not provided
            coregister_id=order.get("coregister_id", ""),
            product_bundle="analytic_udm2",
            continue_existing=order.get("continue_existing", False),
            month_filter=order.get("month_filter", None),
        )
        if "tools" in order:
            task_kwargs["tools"] = (
                set(order["tools"])
                if not isinstance(order["tools"], set)
                else order["tools"]
            )

        task = download_order_by_name(**task_kwargs)
        tasks.append(task)

    # Run all download tasks concurrently with progress tracking
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Downloading Orders")


def filter_items_by_area(
    roi_gdf: gpd.GeoDataFrame, items: List[dict], min_area_percent: float = 0.5
) -> List[dict]:
    """
    Filters a list of items based on their area within a region of interest (ROI). Any items whose area is lower than the minimum area percentage of the ROI will be removed.

    Args:
        roi_gdf (gpd.GeoDataFrame): A GeoDataFrame representing the region of interest.
        items (List[dict]): A list of dictionaries representing the items.
        min_area_percent (float, optional): The minimum area percentage of the ROI that an item must have to be included. Defaults to 0.5.
    Returns:
        List[dict]: A filtered list of items that meet the area criteria.
    """
    # get the ids of each of the items
    ids = [item["id"] for item in items]
    # make a GeoDataFrame from the items
    polygons = [geometry.Polygon(item["geometry"]["coordinates"][0]) for item in items]
    items_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    items_gdf["id"] = ids
    # load the ROI and estimate the UTM CRS
    utm_crs = roi_gdf.estimate_utm_crs()
    # clip the polygons of the images to download to the ROI
    items_gdf = items_gdf.to_crs(utm_crs)
    roi_gdf = roi_gdf.to_crs(utm_crs)
    clipped_gdf = gpd.clip(items_gdf, roi_gdf)

    # calculate the area of the clipped images & ROI
    clipped_gdf["area"] = clipped_gdf["geometry"].area
    roi_gdf["area"] = roi_gdf["geometry"].area
    roi_area = roi_gdf["area"].sum()

    # drop any rows in clipped_gdf whose ROI is less than 50% of the total ROI
    clipped_gdf = clipped_gdf[clipped_gdf["area"] > min_area_percent * roi_area]
    print(f"Filtered out {len(items_gdf) - len(clipped_gdf)} items")
    # filter the items_list to only include the items that are in the clipped_gdf
    item_list = [item for item in items if item["id"] in clipped_gdf["id"].values]
    return item_list


def move_contents(main_folder, psscene_path, remove_path=False):
    """Move all contents of the PSScene directory to the main folder."""
    # Move all contents of PSScene to the main folder
    for item in os.listdir(psscene_path):
        item_path = os.path.join(psscene_path, item)
        shutil.move(item_path, main_folder)

    # Optionally, remove the PSScene directory if it is now empty
    if remove_path:
        os.rmdir(psscene_path)


def move_psscene_contents(main_folder, remove_subdirs=False):
    """Use this function to move all the contents of the PSScene directory to the main folder.
    This is because large planet orders are broken into smaller sub orders and stored in subdirectories.
    """
    # Traverse the main folder to find all subdirectories
    for subdir in os.listdir(main_folder):
        subdir_path = os.path.join(main_folder, subdir)
        if os.path.isdir(subdir_path):
            if os.path.basename(subdir_path) == "PSScene":
                move_contents(main_folder, subdir_path, remove_subdirs)
            # Check if PSScene directory exists within the subdirectory
            psscene_path = os.path.join(subdir_path, "PSScene")
            if os.path.isdir(psscene_path):
                move_contents(main_folder, psscene_path, remove_subdirs)

                # Optionally, remove the subdirectory if requested
            if remove_subdirs:
                if os.path.isdir(subdir_path):
                    shutil.rmtree(subdir_path)


def download_topobathy(
    site: str,
    gdf: gpd.GeoDataFrame,
    save_dir: str,
    doplot: Optional[bool] = True,
    pts_per_deg: int = 300,
    input_size: int = 3,
) -> str:
    """
    Downloads topobathy data, processes it, and optionally generates plots.

    Parameters:
    - site (str): The name of the site.
    - gdf (gpd.GeoDataFrame): The GeoDataFrame containing the site bounds.
    - save_dir (str): The directory where the raster file should be saved.
    - doplot (Optional[bool]): Whether to generate plots (default is True).
    - pts_per_deg (int): Points per degree for resolution (default is 300).
    - input_size (int): Input size for resizing (default is 3).

    Returns:
    - str: The path to the saved or existing raster file.
    """

    # attempt to import bathyreq and if it fails tell user to install it
    try:
        import bathyreq
    except ImportError:
        raise ImportError(
            "bathyreq is not installed. Please install it by running 'pip install bathyreq'"
        )

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    raster_path = os.path.join(save_dir, f"{site}_topobathy.tif")

    # Check if the file already exists
    if os.path.exists(raster_path):
        print(f"File already exists: {raster_path}")
        return raster_path

    # Read the GeoJSON file
    # gdf = gpd.read_file(geofile, driver='GeoJSON')

    # Extract bounds
    minlon = float(gdf.bounds.minx.values[0])
    maxlon = float(gdf.bounds.maxx.values[0])
    minlat = float(gdf.bounds.miny.values[0])
    maxlat = float(gdf.bounds.maxy.values[0])

    # Calculate width and height for the BathyRequest
    kwds = {
        "width": int(np.ceil((maxlon - minlon) * pts_per_deg)),
        "height": int(np.ceil((maxlat - minlat) * pts_per_deg)),
    }
    kwds["width"] = 1000
    kwds["height"] = 1000

    # Request bathymetry data
    req = bathyreq.BathyRequest()
    data, lonvec, latvec = req.get_area(
        longitude=[minlon, maxlon],
        latitude=[minlat, maxlat],
        size=[kwds["width"], kwds["height"]],
    )

    # Clip data to extents
    iy = np.where((lonvec >= minlon) & (lonvec < maxlon))[0]
    ix = np.where((latvec >= minlat) & (latvec < maxlat))[0]
    latvec = latvec[ix]
    lonvec = lonvec[iy]
    data = data[ix[0] : ix[-1], iy[0] : iy[-1]]
    data = np.flipud(data)

    # Resample data
    datar = resize(data, (input_size, input_size), anti_aliasing=True)
    mask = (datar < 30) & (datar > -30)

    # Plot the mask
    if doplot:
        plt.imshow(mask)
        plt.savefig(
            os.path.join(save_dir, f"{site}_10m_mask.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Create custom colormap
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
    colors = np.vstack((colors_undersea, colors_land))
    cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cut_terrain", colors
    )
    norm = FixPointNormalize(sealevel=0, vmax=100)

    if doplot:
        plt.subplot(111, aspect="equal")
        plt.pcolormesh(lonvec, latvec, data, cmap=cut_terrain_map, norm=norm)
        plt.colorbar(extend="both")
        plt.savefig(
            os.path.join(
                save_dir, f"{site}_topobathy_{minlon}_{maxlon}_{minlat}_{maxlat}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save data to GeoTIFF
    xres = (maxlon - minlon) / data.shape[1]
    yres = (maxlat - minlat) / data.shape[0]
    transform = Affine.translation(minlon - xres / 2, minlat - yres / 2) * Affine.scale(
        xres, yres
    )

    with rasterio.open(
        raster_path,
        mode="w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs",
        transform=transform,
    ) as new_dataset:
        new_dataset.write(data, 1)
    print(f"Saving to {os.path.abspath(raster_path)}")

    return raster_path


async def wait_with_exponential_backoff(
    client, order_id, state=None, initial_delay=5, max_attempts=200, callback=None
):
    """
    Wait until the order reaches the desired state using exponential backoff.

    Args:
        client (planet.Client): The Planet client used to poll the order state.
        order_id (str): The ID of the order.
        state (str, optional): State prior to a final state that will end polling. Defaults to None.
        initial_delay (int, optional): Initial delay between polls in seconds. Defaults to 5.
        max_attempts (int, optional): Maximum number of polls. Set to zero for no limit. Defaults to 200.
        callback (Callable[[str], NoneType], optional): Function that handles state progress updates. Defaults to None.

    Returns:
        str: The state of the order on the last poll.
    """
    attempt = 0
    delay = initial_delay

    while True:
        order_state = await client.get_order(order_id)
        current_state = order_state["state"]

        if callback:
            callback(current_state)

        if state and current_state == state:
            return current_state
        if current_state in ["success", "failed", "cancelled"]:
            return current_state

        if max_attempts > 0 and attempt >= max_attempts:
            raise RuntimeError(f"Maximum attempts reached: {max_attempts}")

        attempt += 1
        await asyncio.sleep(delay)
        delay = min(
            delay * 2, 180
        )  # Cap the delay to a maximum of 180 seconds (3 minutes)


class FixPointNormalize(matplotlib.colors.Normalize):
    """
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue/turquise range.
    """

    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val=0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


async def download_order_by_name(
    order_name: str,
    output_path: str,
    roi: gpd.GeoDataFrame,
    start_date: str = "",
    end_date: str = "",
    overwrite: bool = False,
    continue_existing: bool = False,
    order_states: list = None,
    product_bundle: str = "analytic_udm2",
    coregister_id: str = "",
    min_area_percentage: float = 0.5,
    tools: set = None,
    **kwargs,
):
    """
    Downloads an order by name from the Planet API.

    Args:
        order_name (str): The name of the order to download.
        output_path (str): The path where the downloaded order will be saved.
        roi (gpd.geodatafram, optional): The region of interest for the order. Must have a CRS and it should contain a SINGLE ROI polygon.
        start_date (str, optional): The start date for the order. Defaults to an empty string.
        end_date (str, optional): The end date for the order. Defaults to an empty string.
        overwrite (bool, optional): Whether to overwrite an existing order with the same name. Defaults to False.
        order_states (list, optional): The list of states of the order. Defaults to ['success', 'running'].
        product_bundle (str, optional): The product bundle to download. Defaults to "ortho_analytic_4b".
        coregister_id (str, optional): The ID of the image to coregister with. Defaults to "".
        min_area_percentage (float, optional): The minimum area percentage of the ROI's area that must be covered by the images to be downloaded. Defaults to 0.5.
        tools (set, optional): A set of tools to be applied to the order. Defaults to None.
            Example : {"clip", "toar", "coregister"}
            "clip"  If passed, the images will be clipped to the ROI.
            "toar"  If passed, the images will be converted to TOAR reflectance.
            "coregister"  If passed, the images will be coregistered.

    kwargs:
        cloud_cover (float): The maximum cloud cover percentage. (0-1) Defaults to 0.99.

    Raises:
        ValueError: If start_date and end_date are not specified when creating a new order.
        ValueError: If roi is not specified when creating a new order.

    Returns:
        None
    """

    if order_states is None or order_states == []:
        order_states = ["success", "running"]
    elif isinstance(order_states, str):
        order_states = [order_states]

    if overwrite and continue_existing:
        raise ValueError(
            "overwrite and continue_existing cannot both be True. As an order cannot be overwritten and continued at the same time. Please set one to False"
        )

    async with planet.Session() as sess:
        cl = sess.client("orders")

        # check if an existing order with the same name exists
        order_ids = await get_order_ids_by_name(cl, order_name, states=order_states)
        print(f"Matching Order ids: {order_ids}")
        if order_ids != [] and not overwrite:
            print(f"Order with name {order_name} already exists. Downloading...")
            await get_multiple_existing_orders(
                cl, order_ids, output_path, continue_existing=continue_existing
            )
        else:
            action = "Overwriting" if overwrite else "Creating"
            print(f"{action} order with name {order_name}")
            if end_date == "" or start_date == "":
                raise ValueError(
                    "start_date and end_date must be specified to create a new order"
                )
            # if ROI is a geodataframe check if its empty or doesn't have a CRS
            if isinstance(roi, gpd.GeoDataFrame):
                if roi.empty:
                    raise ValueError(
                        "GeoDataFrame roi must not be empty to create a new order"
                    )
                if not roi.crs:
                    raise ValueError("ROI must have a CRS to create a new order")
            if isinstance(roi, dict):
                if not roi:
                    raise ValueError("ROI must not be empty to create a new order")
                roi = gpd.GeoDataFrame.from_features(roi)
                # assume the ROI is in epsg 4326
                roi.set_crs("EPSG:4326", inplace=True)
                print(
                    f"Setting the CRS of the ROI to EPSG:4326. If your ROI is not in this CRS please pass a geodataframe with the correct CRS"
                )

            await make_order_and_download(
                roi,
                start_date,
                end_date,
                order_name,
                output_path,
                product_bundle=product_bundle,
                coregister_id=coregister_id,
                min_area_percentage=min_area_percentage,
                tools=tools,
                **kwargs,
            )


def filter_ids_by_date(items: List[dict], month_filter: List[str]) -> List[str]:
    """
    Get a dictionary of Image IDs grouped based on the acquired date of the items.
    These ids are ordered by acquired date.

    args:
        items (list): A list of items.
        month_filter (list): A list of months to filter the acquired dates by.

    returns:
        dict: A dictionary of Image IDs grouped by acquired date.
        ex. {"2023-06-27":[1234a,33445g],"2023-06-28":[2345c,4f465c]}
    """
    if not month_filter:
        month_filter = [str(i).zfill(2) for i in range(1, 13)]

    ids_by_date = get_ids_by_date(items)

    # Filter and sort the dictionary by date
    filtered_sorted_ids = {
        date: ids_by_date[date]
        for date in sorted(ids_by_date.keys())
        if date.split("-")[1] in month_filter
    }
    return filtered_sorted_ids


def get_ids(items, month_filter: list = None) -> List[str]:
    """
    Get a 1D list of Image IDs grouped based on the acquired date of the items.
    These ids are ordered by acquired date.

    For example, if the items are:
    [
        {"id": 1, "properties": {"acquired": "2023-06-27"}},
        {"id": 2, "properties": {"acquired": "2023-06-28"}},
        {"id": 3, "properties": {"acquired": "2023-06-27"}},
        {"id": 4, "properties": {"acquired": "2023-06-28"}},
    ]

    The output would be [1, 3,2, 4].

    Args:
        items (list): A list of items.

    Returns:
        list: A list of Image IDs.

    """
    if not month_filter:
        month_filter = [str(i).zfill(2) for i in range(1, 13)]

    if items == [] or items is None:
        return []
    # get a dict of each date in format 'YYYY-MM-DD' where each entry is the list of IDS that match this date
    ids_by_date = filter_ids_by_date(items, month_filter)
    # Get the IDs assoicated with each date from the dictionary. This is a nested list ex. [[1,2],[3,4]]
    ids = [ids_by_date.values()]
    # flattens the nested list into a single list ex. [[1,2],[3,4]] -> [1,2,3,4]
    ids = [j for id in ids for j in id]

    # flatten the list of lists into a single list
    ids = [item for sublist in ids for item in sublist]

    return ids


def get_image_id_with_lowest_cloud_cover(items):
    # Ensure the list is not empty
    print(f"items: {items}")
    if not items:
        return None

    # Initialize the variables to store the minimum cloud cover and corresponding image ID
    min_cloud_cover = float("inf")
    image_id_with_lowest_cloud_cover = None

    # Iterate over the items to find the minimum cloud cover
    for item in items:
        if (
            "properties" in item
            and "cloud_cover" in item["properties"]
            and "id" in item
        ):
            cloud_cover = item["properties"]["cloud_cover"]
            if cloud_cover < min_cloud_cover:
                min_cloud_cover = cloud_cover
                image_id_with_lowest_cloud_cover = item["id"]
    return image_id_with_lowest_cloud_cover


def create_combined_filter(
    roi: str,
    time1: str,
    time2: str,
    cloud_cover: float = 0.99,
    product_bundles="basic_analytic_4b",
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a combined filter for downloading planet imagery.

    Args:
        roi (str): Path to geojson with bounds for imagery.
        time1 (str): Start time in the format YYYY-MM-DD.
        time2 (str): End time in the format YYYY-MM-DD.
        cloud_cover (float): The maximum cloud cover percentage. (0-1) Defaults to 0.99.

    Returns:
        dict: Request JSON to download planet imagery.
    """
    ##Get time strings
    day_min = int(time1[-2:])
    month_min = int(time1[-5:-3])
    year_min = int(time1[0:4])
    day_max = int(time2[-2:])
    month_max = int(time2[-5:-3])
    year_max = int(time2[0:4])

    if isinstance(product_bundles, str):
        product_bundles = [product_bundles]
    elif not isinstance(product_bundles, list):
        raise ValueError("product_bundles must be a string or a list of strings")

    analytic_filter = data_filter.asset_filter(product_bundles)
    data_range_filter = data_filter.date_range_filter(
        "acquired",
        datetime(month=month_min, day=day_min, year=year_min),
        datetime(month=month_max, day=day_max, year=year_max),
    )

    print(
        f"Getting data from {time1} to {time2} with cloud cover less than {cloud_cover}"
    )

    cloud_cover_filter = data_filter.range_filter(
        "cloud_cover",
        lte=cloud_cover,
    )
    geom_filter = data_filter.geometry_filter(roi)
    # combining aoi and time and clear percent filter
    combined_filter = data_filter.and_filter(
        [geom_filter, data_range_filter, cloud_cover_filter, analytic_filter]
    )

    return combined_filter


def get_acquired_date(item: dict):
    """
    Get the acquired date from the given item.

    Args:
        item (dict): The item containing the acquired date.

    Returns:
        str: The acquired date in the format 'YYYY-MM-DD'.
    """
    return item["properties"]["acquired"].split("T")[0]


def get_date_item_ids(date, all_items):
    """
    Get the item IDs for items with a specific acquired date.
    Args:
        date (str): The target acquired date in string format (e.g., '2023-06-27').
        all_items (list): A list of item dictionaries, each containing an 'id' field and 'acquired' field.
    Returns:
        list: A list of item IDs that have the specified acquired date.
    """
    return [i["id"] for i in all_items if get_acquired_date(i) == date]


def get_ids_by_date(items):
    """
    Returns a dictionary mapping of acquired dates of the Image IDs to lists of item IDs.
    Args:
        items (list): A list of items.
    Returns:
        dict: A dictionary where the keys are acquired dates and the values are lists of item IDs.
    """
    acquired_dates = [get_acquired_date(item) for item in items]
    unique_acquired_dates = set(acquired_dates)
    # creates a dictionary where the keys are acquired dates and the values are lists of item IDs
    ids_by_date = dict(
        (unique_date, get_date_item_ids(unique_date, items))
        for unique_date in unique_acquired_dates
    )
    return ids_by_date


async def create_and_download(client, request, download_path: str):
    """
    Creates an order using the provided client and request, and then downloads the order to the specified download path.

    Args:
        client (planet.Client): The Planet client used to create the order and download the data.
        request (dict): The request object used to create the order.
        download_path (str): The directory where the downloaded data will be saved.

    Raises:
        planet.exceptions.BadQuery: If the request is invalid.
            Example: {"field":null,"general":[{"message":"Unable to accept order: Your organization does not have permission to run the 'clip' tool"}]}

    Returns:
        dict: The order details.
    """
    # First create the order and wait for it to be created
    with planet.reporting.StateBar(state="creating") as reporter:
        order = await client.create_order(request)
        reporter.update(state="created", order_id=order["id"])
        await wait_with_exponential_backoff(
            client, order["id"], callback=reporter.update_state
        )

    # Download the order to the specified directory
    await client.download_order(order["id"], download_path, progress_bar=True)
    return order


async def get_order_ids_by_name(
    client, order_name: str, states: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """
    Retrieves the order IDs by their name or regex pattern and state(s).

    Args:
        client: The client object used to interact with the API.
        order_name (str): The name or regex pattern of the order name to search for.
        states (list, optional): The list of states of the order. Defaults to ['success'].

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the ID and state of the matching orders.
    """
    if states is None:
        states = ["success"]

    # returns a list of dictionaries containing the order details
    orders_list = await collect(client.list_orders())
    matching_orders = []

    if not orders_list:
        print("No orders found")
        return matching_orders

    print(f"Found {len(orders_list)} orders")

    # First, try to find exact matches
    for order in orders_list:
        if "name" not in order or "state" not in order:
            continue
        if order["name"] == order_name and order["state"] in states:
            matching_orders.append(order["id"])

    # Use regex matching
    order_regex = f"^{order_name}(_batch.*)?$"
    pattern = re.compile(order_regex)

    for order in orders_list:
        if "name" not in order or "state" not in order:
            continue
        if pattern.match(order["name"]) and order["state"] in states:
            matching_orders.append(order["id"])

    # remove duplicate order ids
    matching_orders = list(set(matching_orders))

    if not matching_orders:
        print(f"Order not found with name or pattern {order_name} and states {states}")

    return matching_orders


def validate_order_downloaded(download_path: str) -> bool:
    """
    Validates that the order has been downloaded successfully.

    Args:
        download_path (str): The path to the downloaded order.

    Returns:
        bool: True if the order is downloaded successfully, False otherwise.
    """
    required_dir = "PSScene"
    required_files = [".tif", ".xml", ".json"]

    if not os.path.isdir(download_path):
        return False

    has_required_dir = False
    has_required_files = {ext: False for ext in required_files}

    for root, dirs, files in os.walk(download_path):
        if required_dir in dirs:
            has_required_dir = True
        for file in files:
            for ext in required_files:
                if file.endswith(ext):
                    has_required_files[ext] = True

    return has_required_dir and all(has_required_files.values())


def read_config(config_file_path: str):
    """
    Reads the configuration file and returns a ConfigParser object.

    Parameters:
    config_file_path (str): The path to the configuration file.

    Returns:
    config (ConfigParser): The ConfigParser object containing the configuration data.

    Raises:
    FileNotFoundError: If the configuration file is not found at the specified path.
    """
    # read the api key from the config file
    config = configparser.ConfigParser()
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}")
    config.read(config_file_path)
    return config


def get_ids_by_dates(items):
    """
    Returns a dictionary mapping of acquired dates of the Image IDs to lists of item IDs.
    Args:
        items (list): A list of items.
    Returns:
        dict: A dictionary where the keys are acquired dates and the values are lists of item IDs.
    """
    acquired_dates = [get_acquired_date(item) for item in items]
    unique_acquired_dates = set(acquired_dates)
    ids_by_date = dict((d, get_date_item_ids(d, items)) for d in unique_acquired_dates)
    return ids_by_date


def get_tools(
    roi_path: str = "",
    clip: bool = True,
    toar: bool = True,
    coregister: bool = False,
    id_to_coregister: str = "",
):
    """
    Returns a list of tools based on the provided parameters.

    Args:
        roi_path (str, optional): Path to the ROI file. Defaults to "".
        clip (bool, optional): Flag indicating whether to perform clipping. Defaults to True.
        toar (bool, optional): Flag indicating whether to perform TOAR (Top of Atmosphere Reflectance) conversion. Defaults to True.
        coregister (bool, optional): Flag indicating whether to perform coregistration. Defaults to False.
        id_to_coregister (str, optional): ID of the image to coregister with. Defaults to "".

    Returns:
        list: List of tools based on the provided parameters.
    """
    tools = []
    if clip and roi_path:
        tools.append(planet.order_request.clip_tool(aoi=roi_path))
    if toar:
        tools.append(planet.order_request.toar_tool(scale_factor=10000))
    if coregister and id_to_coregister:
        tools.append(planet.order_request.coregister_tool(id_to_coregister))
    return tools


async def get_item_list(roi, start_date, end_date, **kwargs):
    """
    Get a list of item IDs based on the provided parameters.

    KwArgs:
        cloud_cover (float): The maximum cloud cover percentage. (0-1) Defaults to 0.99.
        month_filter (list): A list of months to filter the acquired dates by.
            Pass a list of months in the format ['01','02','03'] to filter the acquired dates by the months in the list.
            Defaults to an empty list which means all months are included.

    """
    defaults = {
        "cloud_cover": 0.99,
    }
    defaults.update(kwargs)

    async with planet.Session() as sess:
        cl = sess.client("data")

        combined_filter = create_combined_filter(roi, start_date, end_date, **defaults)

        # Create the order request

        request = await cl.create_search(
            name="temp_search", search_filter=combined_filter, item_types=["PSScene"]
        )
        # stats = await cl.get_stats(item_types=["PSScene"], interval="day",search_filter=combined_filter)
        # print(stats)
        # 100,000 is the highest limit for the search request that has been tested
        # create a search request that returns 100 items per page see this for an example https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/Data-API/planet_python_client_introduction.ipynb
        items = cl.run_search(search_id=request["id"], limit=10000)
        item_list = [i async for i in items]

        if item_list == []:
            print(
                f"No items found that matched the search criteria were found.\n\
                  start_date: {start_date}\n\
                  end_date: {end_date}\n\
                  cloud_cover: {kwargs.get('cloud_cover', 0.99)}"
            )

        # print(f"items: {item_list}")
        # get the ids of the items group by acquired date, then flattened into a 1D list.
        if "month_filter" in kwargs:
            ids = get_ids(item_list, kwargs["month_filter"])
        else:
            ids = get_ids(item_list)
        return ids


async def order_by_ids(
    roi,
    ids,
    order_name,
    download_path: str,
    clip: bool = True,
    toar: bool = True,
    coregister: bool = False,
):

    async with planet.Session() as sess:
        # create a client for the orders API
        cl = sess.client("orders")

        # get the tools to be applied to the order
        tools = get_tools(roi, clip, toar, coregister, ids[0])
        # analytic_udm2
        # By default use the clip and TOAR tools to clip the image to the roi and convert the images from radience to TOAR reflectance

        request = planet.order_request.build_request(
            name=order_name,
            products=[
                planet.order_request.product(
                    item_ids=ids,
                    product_bundle="ortho_analytic_4b",
                    item_type="PSScene",
                )
            ],
            tools=tools,
        )

        # Create and download the order
        order = await create_and_download(cl, request, download_path)


async def process_order_batch(
    cl,
    ids_batch,
    tools,
    order_name_base,
    download_path,
    product_bundle="ortho_analytic_4b",
):
    order_name = f"{order_name_base}_{len(ids_batch)}"
    request = planet.order_request.build_request(
        name=order_name,
        products=[
            planet.order_request.product(
                item_ids=ids_batch, product_bundle=product_bundle, item_type="PSScene"
            )
        ],
        tools=tools,
    )

    # Create and download the order
    await create_and_download(cl, request, download_path)


async def process_orders_in_batches(
    cl,
    ids: List[str],
    tools: List,
    download_path,
    order_name_base,
    product_bundle="ortho_analytic_4b",
    id_to_coregister: str = "",
):
    """
    Process orders in batches of 499 or less.
    Planet API has a limit of 500 items per order. This function splits the ids list into batches of 499 or less and processes each batch in parallel.
    Args:
        cl: The Planet API orders client.
        ids: A list of item IDs to process.
        tools: A list of tools to apply to the order.
            List of planet tools to apply to the order created with the planet planet.order_request.<tool_name> function.

        download_path: The path to the download directory.
        order_name_base: The base name for the order.
        product_bundle: The product bundle to use for the order.
        id_to_coregister: The ID of the image to coregister with (optional).

    """
    # Split the ids list into batches of 499 or less
    if id_to_coregister:
        print(f"adding the id to coregister {id_to_coregister} to each batch of ids")
        # add the id_to_coregister to each batch of ids
        id_batches = [
            ids[i : i + 499] + [id_to_coregister] for i in range(0, len(ids), 499)
        ]
    else:
        id_batches = [ids[i : i + 499] for i in range(0, len(ids), 499)]

    # Create tasks for each batch
    tasks = []
    for i, ids_batch in enumerate(id_batches):
        print(
            f"processing batch #{i + 1} of size {len(ids_batch)} of {len(id_batches)}"
        )
        task = process_order_batch(
            cl,
            ids_batch,
            tools,
            f"{order_name_base}_batch_{i + 1}",
            download_path,
            product_bundle,
        )
        tasks.append(task)

    # Download orders in parallel, no more than 5 at a time
    semaphore = asyncio.Semaphore(5)

    async def limited_parallel_execution(task):
        async with semaphore:
            await task

    await asyncio.gather(*(limited_parallel_execution(task) for task in tasks))


def build_tools_list(tools, roi_dict=None, id_to_coregister: str = ""):
    """
    Constructs a list of tool requests based on the provided tool names designed to work with
    the Planet Labs API.

    Args:
        tools (list): A list of tool names (str).
            Available tools: "clip", "toar", "coregister".
        roi_dict (dict): clip GeoJSON, either Polygon or Multipolygon.
            Example: {"type": "Polygon", "coordinates": [[[lon1, lat1], [lon2, lat2], ...]]}
        id_to_coregister (str, optional): The ID of the image to coregister the order with. Defaults to ""
            This assumes that the id is already in the list of ids to be downloaded.
            If not, it will raise an error.
    Returns:
        list: A list of tool request objects.
    """
    tool_map = {
        "clip": lambda: planet.order_request.clip_tool(aoi=roi_dict),
        "toar": lambda: planet.order_request.toar_tool(scale_factor=10000),
        "coregister": lambda: planet.order_request.coregister_tool(id_to_coregister),
    }

    tools_list = []
    for raw_tool in tools:
        tool_name = raw_tool.strip().lower()
        if tool_name in tool_map:
            tools_list.append(tool_map[tool_name]())
        else:
            print(f"Warning: Unknown tool '{raw_tool}' skipped.")

    print(f"Using the following tools list: {tools_list}")
    return tools_list


def get_id_to_coregister(ids, items, coregister: bool, user_coregister_id: str = ""):
    """
    If coregistration is requested, this function checks if the user-supplied coregister ID is in the list of IDs to download.
    If not, it raises a ValueError. If the user-supplied ID is empty, it selects the image with the lowest cloud cover from the list of items.
    Args:
        ids (list): List of item IDs to be downloaded.
        items (list): List of item dicts (with properties).
        coregister (bool): Whether coregistration is requested.
        user_coregister_id (str, optional): User-supplied coregister ID. Defaults to "".
    Returns:
        str: The ID to use for coregistration, or "" if not needed.
    Raises:
        ValueError: If coregistration is requested but the ID is not in the list.
    """
    if not coregister:
        return ""
    id_to_coregister = user_coregister_id or get_image_id_with_lowest_cloud_cover(items)
    if id_to_coregister not in ids:
        raise ValueError(
            f"Coregister ID {id_to_coregister} not found in the list of IDs to download"
        )
    return id_to_coregister


async def make_order_and_download(
    roi,
    start_date,
    end_date,
    order_name,
    download_path: str,
    coregister_id: str = "",
    product_bundle: str = "ortho_analytic_4b",
    min_area_percentage: float = 0.5,
    tools: set[str] = None,
    **kwargs,
):
    """
    Creates an order request for downloading satellite images from Planet API based on the given parameters and downloads the images to the specified output folder.

    Args:
        roi_path (str): The path to the region of interest (ROI) file.
        start_date (str): The start time of the acquisition period for the satellite images.
        end_date (str): The end time of the acquisition period for the satellite images.
        order_name (str): The name of the order.
        output_folder (str): The folder where the downloaded images will be saved.
        coregister_id (str, optional): The ID of the image to coregister with. Defaults to "".
        product_bundle (str, optional): The product bundle to download. Defaults to "ortho_analytic_4b".
        min_area_percentage (float, optional): The minimum area percentage of the ROI's area that must be covered by the images to be downloaded. Defaults to 0.5.
        tools (set[str], optional): A set of tools to be applied to the order. Defaults to an empty set.
            Example: {"clip", "toar", "coregister"}
            clip (bool, optional): Whether to clip the images to the ROI. Defaults to True.
            toar (bool, optional): Whether to convert the images to TOAR reflectance. Defaults to True.
            coregister (bool, optional): Whether to coregister the images. Defaults to False.
    kwargs:
        cloud_cover (float): The maximum cloud cover percentage. (0-1) Defaults to 0.99.

    Returns:
        None
    """
    if tools is None:
        tools = set()

    # Ensure cloud_cover is in kwargs with a default value of 0.99
    kwargs.setdefault("cloud_cover", 0.99)

    async with planet.Session() as sess:

        # convert the ROI to a dictionary so it can be used with the Planet API
        roi_dict = json.loads(roi.to_json())

        # This queries the data api and returns a list of items (whole tiles) that intersect the roi and match date range
        item_list = await query_planet_items(
            sess, roi_dict, start_date, end_date, **kwargs
        )

        # filter the items list by area. If the area of the image is less than than percentage of area of the roi provided, then the image is not included in the list
        print(f"Number of items to download before filtering by area: {len(item_list)}")
        item_list = filter_items_by_area(roi, item_list, min_area_percentage)

        print(f"Number of items to download after filtering by area: {len(item_list)}")
        # gets the IDs of the items to be downloaded based on the acquired date (not order by date though)
        month_filter = kwargs.get("month_filter")
        if month_filter:
            print(f"Applying Month filter: {month_filter}")
        ids = get_ids(item_list, month_filter)

        print(f"Requesting {len(ids)} items")

        # Get the ID to coregister with
        id_to_coregister = get_id_to_coregister(
            ids, item_list, "coregister" in tools, coregister_id
        )

        # create a client for the orders API
        order_client = sess.client("orders")

        # If tools set/list is provided, use it to build the tools list, else use booleans
        tools_list = build_tools_list(
            tools, roi_dict=roi_dict, id_to_coregister=id_to_coregister
        )

        # print(f"id_to_coregister: {id_to_coregister} Apply Coregistration: {coregister}")
        # Process orders in batches
        print(f"Total number of scenes to download: {len(ids)}")
        await process_orders_in_batches(
            order_client,
            ids,
            tools_list,
            download_path,
            order_name,
            product_bundle=product_bundle,
            id_to_coregister=id_to_coregister,
        )


async def get_existing_order(
    client, order_id: str, download_path="downloads", continue_existing=False
) -> Optional[dict]:
    """
    Downloads the contents of an order from the client.

    Args:
        client: The client object used to interact with the API.
        order_id: The ID of the order to download.
        download_path: The path where the downloaded files will be saved. Defaults to 'downloads'.
        continue_existing: Flag indicating whether to continue downloading an existing order. Defaults to False.
    Returns:
        Optional[dict]: The order if successful, None otherwise.
    """
    order = await client.get_order(order_id)
    print(f"The order's state is {order['state']}")
    if order["state"] == "success":
        if continue_existing or not validate_order_downloaded(download_path):
            print(
                f"The order is ready to download. Downloading the order to {download_path}"
            )
            await client.download_order(order["id"], download_path, progress_bar=True)
        else:
            print(f"Order already downloaded to {download_path}")
            return order
    elif order["state"] == "running":
        print(f"Order is running and downloading to {download_path}")
        client.download_order(order["id"], download_path, progress_bar=True)
    else:
        print("Order is not yet fulfilled.")
        return None


async def get_multiple_existing_orders(
    client,
    order_ids,
    download_path="downloads",
    continue_existing=False,
    max_concurrent_downloads=5,
):
    """
    Downloads the contents of multiple orders from the client in parallel with a limit on concurrent downloads.

    Args:
        client: The client object used to interact with the API.
        order_ids: The list of order IDs to download.
        download_path: The path where the downloaded files will be saved. Defaults to 'downloads'.
        continue_existing: Flag indicating whether to continue downloading existing orders. Defaults to False.
        max_concurrent_downloads: The maximum number of concurrent downloads. Defaults to 5.
    Returns:
        List[Optional[dict]]: A list of orders if successful, None otherwise.
    """
    semaphore = asyncio.Semaphore(max_concurrent_downloads)

    async def download_order_with_semaphore(order_id):
        async with semaphore:
            return await get_existing_order(
                client, order_id, download_path, continue_existing
            )

    download_tasks = [download_order_with_semaphore(order_id) for order_id in order_ids]
    return await asyncio.gather(*download_tasks)


# get the order ids
async def cancel_order_by_name(
    order_name: str,
    order_states: list = None,
):
    """
    Cancels an order by its name.

    Args:
        order_name (str): The name of the order to cancel.
        order_states (list, optional): A list of order states to filter the search. Defaults to None.

    Returns:
        None
    """
    async with planet.Session() as sess:
        cl = sess.client("orders")
        # check if an existing order with the same name exists
        order_ids = await get_order_ids_by_name(cl, order_name, states=order_states)
        if order_ids == []:
            print(
                f"No order found with name {order_name} was found with states {order_states}"
            )
        print(f"order_ids requested to cancel: {order_ids}")
        canceled_orders_info = await cl.cancel_orders(order_ids)
        print(f"canceled_orders_info: {canceled_orders_info}")
