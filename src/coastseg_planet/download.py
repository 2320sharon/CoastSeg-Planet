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

async def download_order_by_name(order_name:str,
                                 output_path:str,
                                 roi:dict={},
                                 start_date:str="",
                                 end_date:str="",
                                 overwrite:bool=False):
    """
    Downloads an order by name from the Planet API.

    Args:
        order_name (str): The name of the order to download.
        output_path (str): The path where the downloaded order will be saved.
        roi (dict, optional): The region of interest for the order. Defaults to an empty dictionary.
        start_date (str, optional): The start date for the order. Defaults to an empty string.
        end_date (str, optional): The end date for the order. Defaults to an empty string.
        overwrite (bool, optional): Whether to overwrite an existing order with the same name. Defaults to False.

    Raises:
        ValueError: If start_date and end_date are not specified when creating a new order.
        ValueError: If roi is not specified when creating a new order.

    Returns:
        None
    """
    async with planet.Session() as sess:
        cl = sess.client('orders')
        # check if an existing order with the same name exists
        order_id = await get_order_id_by_name(cl, order_name)
        if order_id is not None and not overwrite:
            print(f'Order with name {order_name} already exists')
            await get_existing_order(cl, order_id, output_path)
        # if no order id or overwrite is true, then create a new order & download it
        else:
            action = "Overwriting" if overwrite else "Creating"
            print(f'{action} order with name {order_name}')
            if end_date == "" or start_date == "":
                raise ValueError("start_date and end_date must be specified to create a new order")
            if roi == {}:
                raise ValueError("roi must be specified to create a new order")
            await make_order_and_download(roi, start_date, end_date, order_name, output_path)
    
    

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
    
    data_range_filter = data_filter.date_range_filter("acquired",
                                                      datetime(month=month_min,
                                                               day=day_min,
                                                               year=year_min),
                                                      datetime(month=month_max,
                                                               day=day_max,
                                                               year=year_max))

    geom_filter = data_filter.geometry_filter(roi)
    
    #combining aoi and time and clear percent filter
    combined_filter = data_filter.and_filter([geom_filter,
                                              data_range_filter])
    return combined_filter

def get_acquired_date(item:dict):
    """
    Get the acquired date from the given item.

    Args:
        item (dict): The item containing the acquired date.

    Returns:
        str: The acquired date in the format 'YYYY-MM-DD'.
    """
    return item['properties']['acquired'].split('T')[0]

def get_date_item_ids(date, all_items):
    """
    Get the item IDs for items with a specific acquired date.
    Args:
        date (str): The target acquired date in string format (e.g., '2023-06-27').
        all_items (list): A list of item dictionaries, each containing an 'id' field and 'acquired' field.
    Returns:
        list: A list of item IDs that have the specified acquired date.
    """
    return [i['id'] for i in all_items if get_acquired_date(i) == date]

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
    ids_by_date = dict((d, get_date_item_ids(d, items))
                       for d in unique_acquired_dates)
    return ids_by_date

async def create_and_download(client, request, download_path:str):
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
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(request)
        reporter.update(state='created',
                        order_id=order['id'])
        await client.wait(order['id'],
                          callback=reporter.update_state)
    # download the order to the specified directory
    await client.download_order(order['id'],
                                download_path,
                                progress_bar=True)
    return order

async def get_order_id_by_name(client, order_name, state='success'):
    """
    Retrieves the order ID by its name and state.

    Args:
        client: The client object used to interact with the API.
        order_name (str): The name of the order to search for.
        state (str, optional): The state of the order. Defaults to 'success'.

    Returns:
        str: The ID of the order if found, None otherwise.
    """
    orders_list = await collect(client.list_orders())
    for order in orders_list:
        if order['name'] == order_name and order['state'] == state:
            return order['id']
    print(f'Order not found with name {order_name} and state {state}')
    return None

def validate_order_downloaded(download_path: str) -> bool:
    """
    Validates that the order has been downloaded successfully.

    Args:
        download_path (str): The path to the downloaded order.

    Returns:
        bool: True if the order is downloaded successfully, False otherwise.
    """
    required_dir = 'PSScene'
    required_files = ['.tif', '.xml', '.json']

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


      
def read_config(config_file_path:str):
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
        
async def download_existing_order(order_name:str, config_file_path:str):
    """
    Downloads the contents of an existing order from the Planet API.

    Returns:
        None
    """
    # read the api key from the config file
    config = read_config(config_file_path)
    os.environ['API_KEY'] = config['DEFAULT']['API_KEY']

    auth = Auth.from_env('API_KEY')
    auth.store()

    session = Session(auth=auth)
    client =  OrdersClient(session=session)
    order_id = await get_order_id_by_name(client, order_name)
    # create the path to download the order to the 'downloads' directory
    download_path = os.path.join(os.getcwd(), 'downloads', order_name)

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
    ids_by_date = dict((d, get_date_item_ids(d, items))
                       for d in unique_acquired_dates)
    return ids_by_date

def get_tools(roi_path:str="",clip:bool=True,toar:bool=True,coregister=False,id_to_coregister:str="",):
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

async def make_order_and_download(roi, start_date,
                                  end_date,
                                  order_name, download_path:str, clip: bool = True, toar: bool = True, coregister: bool = False):
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
        cl = sess.client('data')
        
        combined_filter = create_combined_filter(roi, start_date, end_date)

        # Create the order request
        request = await cl.create_search(name='temp_search', search_filter=combined_filter, item_types=['PSScene'])
        
        items = cl.run_search(search_id=request['id'])
        item_list = [i async for i in items]
        # get the ids of the items group by acquired date
        ids = get_ids(item_list)

        # create a client for the orders API
        cl = sess.client('orders')

        # get the tools to be applied to the order
        tools = get_tools(roi, clip, toar, coregister, ids[0])
        # By default use the clip and TOAR tools to clip the image to the roi and convert the images from radience to TOAR reflectance
        request = planet.order_request.build_request(name=order_name,
                                                     products=[planet.order_request.product(item_ids=ids,
                                                                                            product_bundle='analytic_udm2',
                                                                                            item_type='PSScene')],
                                                     tools=tools)
        
        # Create and download the order
        order = await create_and_download(cl, request, download_path)
        
async def get_existing_order(client, order_id, download_path='downloads'):
    """
    Downloads the contents of an order from the client.

    Args:
        client: The client object used to interact with the API.
        order_id: The ID of the order to download.
        download_path: The path where the downloaded files will be saved. Defaults to 'downloads'.

    Returns:
        None
    """
    order = await client.get_order(order_id)
    print(f"The order's state is {order['state']}")
    if order['state'] == 'success':
        if validate_order_downloaded(download_path):
            print(f"Order already downloaded to {download_path}")
            return order
        else:
            print(f"The order is ready to download. Downloading the order to {download_path}")
            await client.download_order(order['id'], download_path, progress_bar=True)
    else:
        print('Order is not yet fulfilled.')
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