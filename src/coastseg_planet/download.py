import os
from planet import collect
from planet import Auth, Session, DataClient, OrdersClient, data_filter, order_request
import planet
from pprint import pprint
import os
import asyncio
from datetime import datetime
from shapely.geometry import MultiPolygon, shape
import configparser
import matplotlib.pyplot as plt
from rasterio.transform import Affine
import matplotlib
import numpy as np
import geopandas as gpd
from typing import Optional
import bathyreq
from skimage.transform import resize
import rasterio
from typing import Optional


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
    roi: dict = {},
    start_date: str = "",
    end_date: str = "",
    overwrite: bool = False,
    continue_existing: bool = False,
    order_states: list = None,
):
    """
    Downloads an order by name from the Planet API.

    Args:
        order_name (str): The name of the order to download.
        output_path (str): The path where the downloaded order will be saved.
        roi (dict, optional): The region of interest for the order. Defaults to an empty dictionary.
        start_date (str, optional): The start date for the order. Defaults to an empty string.
        end_date (str, optional): The end date for the order. Defaults to an empty string.
        overwrite (bool, optional): Whether to overwrite an existing order with the same name. Defaults to False.
        order_states (list, optional): The list of states of the order. Defaults to ['success'].
    Raises:
        ValueError: If start_date and end_date are not specified when creating a new order.
        ValueError: If roi is not specified when creating a new order.

    Returns:
        None
    """
    if order_states is None:
        order_states = ["success", "running"]
    elif isinstance(order_states, str):
        order_states = [order_states]

    async with planet.Session() as sess:
        cl = sess.client("orders")
        # check if an existing order with the same name exists
        order_id, order_state = await get_order_id_by_name(
            cl, order_name, states=order_states
        )
        if order_id is not None and not overwrite:
            print(f"Order with name {order_name} already exists")
            if order_state == "running":
                print(f"Please wait the order is still running")
            await get_existing_order(
                cl, order_id, output_path, continue_existing=continue_existing
            )
        # if no order id or overwrite is true, then create a new order & download it
        else:
            action = "Overwriting" if overwrite else "Creating"
            print(f"{action} order with name {order_name}")
            if end_date == "" or start_date == "":
                raise ValueError(
                    "start_date and end_date must be specified to create a new order"
                )
            if roi == {}:
                raise ValueError("roi must be specified to create a new order")
            await make_order_and_download(
                roi, start_date, end_date, order_name, output_path
            )


def get_ids(items):
    """
    Get a list of Image IDs grouped based on the acquired date of the items.

    Args:
        items (list): A list of items.

    Returns:
        list: A list of Image IDs.

    """
    acquired_dates = [get_acquired_date(item) for item in items]
    unique_acquired_dates = set(acquired_dates)
    ids_by_date = get_ids_by_date(items)
    # list Image IDs grouped based on Acquired Date
    ids = [ids_by_date[j] for j in list(unique_acquired_dates)]
    # flattens the nested list into a single list ex. [[1,2],[3,4]] -> [1,2,3,4]
    ids = [j for id in ids for j in id]
    return ids


def create_combined_filter(roi, time1, time2):
    """
    Create a combined filter for downloading planet imagery.

    Args:
        roi (str): Path to geojson with bounds for imagery.
        time1 (str): Start time in the format YYYY-MM-DD.
        time2 (str): End time in the format YYYY-MM-DD.

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

    analytic_filter = data_filter.asset_filter(["basic_analytic_4b"])
    data_range_filter = data_filter.date_range_filter(
        "acquired",
        datetime(month=month_min, day=day_min, year=year_min),
        datetime(month=month_max, day=day_max, year=year_max),
    )

    geom_filter = data_filter.geometry_filter(roi)

    # instrument_filter = data_filter.string_in_filter("instrument", ["PS2", "PSB.SD","PS2.SD"])

    # combining aoi and time and clear percent filter
    combined_filter = data_filter.and_filter([geom_filter, data_range_filter])

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
    ids_by_date = dict((d, get_date_item_ids(d, items)) for d in unique_acquired_dates)
    return ids_by_date


async def create_and_download(client, request, download_path: str):
    """
    Creates an order using the provided client and request, and then downloads the order to the specified download path.

    Args:
        client (PlanetClient): The Planet client used to create the order and download the data.
        request (dict): The request object used to create the order.
        download_path (str): The directory where the downloaded data will be saved.

    Returns:
        None
    """
    # first create the order and wait for it to be created
    with planet.reporting.StateBar(state="creating") as reporter:
        order = await client.create_order(request)
        reporter.update(state="created", order_id=order["id"])
        await client.wait(order["id"], callback=reporter.update_state)
    # download the order to the specified directory
    await client.download_order(order["id"], download_path, progress_bar=True)
    return order


async def get_order_id_by_name(client, order_name, states=None):
    """
    Retrieves the order ID by its name and state(s).

    Args:
        client: The client object used to interact with the API.
        order_name (str): The name of the order to search for.
        states (list, optional): The list of states of the order. Defaults to ['success'].

    Returns:
        str: The ID of the order if found, None otherwise.
    """
    if states is None:
        states = ["success"]

    orders_list = await collect(client.list_orders())
    for order in orders_list:
        if order["name"] == order_name and order["state"] in states:
            return order["id"], order["state"]

    print(f"Order not found with name {order_name} and states {states}")
    return None, None


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


async def download_existing_order(order_name: str, config_file_path: str):
    """
    Downloads the contents of an existing order from the Planet API.

    Returns:
        None
    """
    # read the api key from the config file
    config = read_config(config_file_path)
    os.environ["API_KEY"] = config["DEFAULT"]["API_KEY"]

    auth = Auth.from_env("API_KEY")
    auth.store()

    session = Session(auth=auth)
    client = OrdersClient(session=session)
    order_id = await get_order_id_by_name(client, order_name)
    # create the path to download the order to the 'downloads' directory
    download_path = os.path.join(os.getcwd(), "downloads", order_name)

    await download_order_contents(client, order_id, download_path)
    return


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
    coregister=False,
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


async def get_item_list(roi, start_date, end_date):

    async with planet.Session() as sess:
        cl = sess.client("data")

        combined_filter = create_combined_filter(roi, start_date, end_date)

        # Create the order request
        request = await cl.create_search(
            name="temp_search", search_filter=combined_filter, item_types=["PSScene"]
        )

        items = cl.run_search(search_id=request["id"])
        item_list = [i async for i in items]
        # get the ids of the items group by acquired date
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
                    item_ids=ids, product_bundle="analytic_udm2", item_type="PSScene"
                )
            ],
            tools=tools,
        )

        # Create and download the order
        order = await create_and_download(cl, request, download_path)


async def make_order_and_download(
    roi,
    start_date,
    end_date,
    order_name,
    download_path: str,
    clip: bool = True,
    toar: bool = True,
    coregister: bool = False,
):
    """
    Creates an order request for downloading satellite images from Planet API based on the given parameters and downloads the images to the specified output folder.

    Args:
        roi_path (str): The path to the region of interest (ROI) file.
        start_date (str): The start time of the acquisition period for the satellite images.
        end_date (str): The end time of the acquisition period for the satellite images.
        order_name (str): The name of the order.
        output_folder (str): The folder where the downloaded images will be saved.
        clip (bool, optional): Whether to clip the images to the ROI. Defaults to True.
        toar (bool, optional): Whether to convert the images to TOAR reflectance. Defaults to True.
        coregister (bool, optional): Whether to coregister the images. Defaults to False.

    Returns:
        None
    """

    async with planet.Session() as sess:
        cl = sess.client("data")

        combined_filter = create_combined_filter(roi, start_date, end_date)

        # Create the order request
        request = await cl.create_search(
            name="temp_search", search_filter=combined_filter, item_types=["PSScene"]
        )

        items = cl.run_search(search_id=request["id"])
        item_list = [i async for i in items]
        # get the ids of the items group by acquired date
        ids = get_ids(item_list)

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
                    item_ids=ids, product_bundle="analytic_udm2", item_type="PSScene"
                )
            ],
            tools=tools,
        )

        # Create and download the order
        order = await create_and_download(cl, request, download_path)


async def get_existing_order(
    client, order_id, download_path="downloads", continue_existing=False
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


# async def download_order_contents(client, order_id, download_path='downloads'):
#     """
#     Downloads the contents of an order from the client.

#     Args:
#         client: The client object used to interact with the API.
#         order_id: The ID of the order to download.
#         download_path: The path where the downloaded files will be saved. Defaults to 'downloads'.

#     Returns:
#         None
#     """
#     order = await client.get_order(order_id)
#     if order['state'] == 'success':
#         if validate_order_downloaded(download_path):
#             print(f"Order already downloaded to {download_path}")
#             return
#         else:
#             print(f"Downloading the order to {download_path}")
#     else:
#         print('Order is not yet fulfilled.')
