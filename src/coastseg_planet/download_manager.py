import asyncio
import planet
import os
import pathlib
import re
from typing import Dict, Any, Union, List, Set, Optional

# import geopandas as gpd
from planet import OrdersClient
import json

# from functools import partial
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from coastseg_planet.config import TILE_STATUSES, CLOUD_COVER, MIN_AREA_PERCENT
from coastseg_planet import download
from coastseg_planet.orders import Order
from coastseg_planet.db import DuckDBInterface
from coastseg_planet.processor import TileProcessor


def extract_unique_datetime_ids(items: Union[str, List[str]]) -> Union[str, Set[str]]:
    """
    Extracts unique datetime IDs in the format 'YYYYMMDD_HHMMSS_XX' from a string or list of strings.

    Args:
        items (Union[str, List[str]]): A filename string or list of filename strings.

    Returns:
        Union[str, Set[str]]: A single datetime ID string if one match is found, otherwise a set of unique IDs.
    """
    datetime_pattern = re.compile(r"\d{8}_\d{6}_\d{2}")
    unique_ids = set()

    # Normalize to a list for uniform handling
    if isinstance(items, str):
        items = [items]

    for filename in items:
        match = datetime_pattern.search(filename)
        if match:
            unique_ids.add(match.group())

    if len(unique_ids) == 1:
        return next(iter(unique_ids))
    return unique_ids


def read_roi_from_order(order: Order):
    """
    Reads the ROI from the order object and returns it as a GeoDataFrame.

    Args:
        order (Order): The order object containing the ROI.

    Returns:
        dict: A dictionary containing the ROI ID and its geometry in GeoJSON format.
    """
    gdf = order.get_roi_geodataframe()
    order_dict = order.to_dict()
    roi_id = order_dict.get("roi_id", "")
    # roi_dict = json.loads(roi_gdf.to_json())
    # Select the first feature (as a GeoDataFrame with one row)
    first_feature_gdf = gdf.iloc[[0]]
    # Convert to a FeatureCollection string
    feature_collection_json = first_feature_gdf.to_json()
    roi_dict = json.loads(feature_collection_json)
    # extract the first feature as a dict
    first_feature = roi_dict["features"][
        0
    ]  # this contains the geometry and properties of the first feature

    roi = {
        "roi_id": roi_id,
        "roi_geometry": first_feature["geometry"],
    }
    return roi


class DownloadManager:
    def __init__(
        self,
        processor: TileProcessor,
        db: DuckDBInterface,
        planet_session: planet.Session,
        client: Optional[OrdersClient] = None,
        order_semaphore: asyncio.Semaphore = asyncio.Semaphore(5),
        download_semaphore: asyncio.Semaphore = asyncio.Semaphore(10),
    ):
        """
        Initializes the DownloadManager with the given parameters.

        Args:
            processor (TileProcessor): The processor to handle tile processing, this handles queuing and processing of the tiles before entering the database.
            db (DuckDBInterface): The database interface for storing metadata about the downloaded tiles.
            planet_session (planet.Session): The Planet session to use for API requests. This is shared across all clients.
            client (planet.Client): The Planet Orders client to use for API requests to the orders API.
            order_semaphore (asyncio.Semaphore): Semaphore for limiting concurrent order creation.
                API requests are limited to 5 requests per second for the activation API as of 5/1/2025
            download_semaphore (asyncio.Semaphore): Semaphore for limiting concurrent downloads.
                API requests are limited to 5 requests per second for the download API as of 5/1/2025


        """
        self.planet_session = planet_session

        self._orders_client = client  # backing attribute
        # if client is None:
        #     self.orders_client = self.create_order_client()
        # else:
        #     self.orders_client = client

        self.processor = processor
        self.db = db
        self.order_semaphore = order_semaphore
        self.download_semaphore = download_semaphore

    def create_order_client(self):
        """
        Uses the existing session to create a new client for the orders API.

        Returns:
            planet.api.Client: A client object for interacting with the Planet API's orders service.
        """
        return self.planet_session.client("orders")

    @property
    def orders_client(self):
        """
        Property to access the orders client. If the client has been closed, recreate it.
        """
        if self._orders_client is None:
            # Recreate the client if not present or underlying HTTP client is closed
            self._orders_client = self.create_order_client()
        return self._orders_client

    async def create_order(self, request: dict) -> dict:
        """
        Asynchronously creates an order using the provided request dictionary.

        Args:
            request (dict): A dictionary containing the details of the order to be created returned by the planet.order_request.build_request

        Returns:
            dict:  JSON description of the created order.

        Raises:
            Exception: If the order creation fails or an error occurs during the process.
        """
        order = await self.orders_client.create_order(request)
        return order

    async def await_order(self, request: dict) -> dict:
        """
        Creates an order using the provided client and request.
        The order is created and the state is updated using exponential backoff.
        Waits for the order to enter one of the following states:
        - cancelled
        - success
        - failed

        Note: once the order enters the "queued" and "running" states this function waits for the order to enter one of the above states.
        This is done using exponential backoff to avoid overwhelming the API with requests.

        Args:
            request (dict): The request object used to create the order.

        Returns:
            dict: The order details.
                Not downloadable. Must call api with orders_client.get_order(order_id) to recieve downloadable version
        """
        # First create the order and wait for it to be created
        with planet.reporting.StateBar(state="creating") as reporter:
            order = await self.orders_client.create_order(request)
            reporter.update(state="created", order_id=order["id"])
            await download.wait_with_exponential_backoff(
                self.orders_client, order["id"], callback=reporter.update_state
            )
        return order

    async def download_orders(
        self, orders: List[Order], order_contains_name: bool = False
    ):
        """
        Processes a list of orders for downloading. Note this implemention only works for order where the clip tool is used.

        For each order, this method:
            - Checks if a matching order already exists in the API based on name and state.
            - If a successful order exists, attempts to download it.
            - If a running order exists, prepares to handle it (logic in progress).
            - If no matching order is found, creates a new order and initiates the download.
            - The `contains_name` flag allows for substring matching when searching for existing orders.

        Args:
            orders (List[Order]): A list of Order objects to process.
            order_contains_name (bool, optional): If True, matches existing orders that contain the name
                rather than exact or pattern match. Defaults to False.

        Note:
            - The full downloading logic and running order handling is still being implemented.
            - Many sections are placeholders or marked for future integration and testing.
        """
        # For each order
        # Check if the order exists at the orders api
        # If it does, get the order id
        # add continue_existing_order to list of tasks ( this means check the DB and then downloading missing files)

        # if it does not create a new order, then download order

        orders_to_download = []

        for order in orders:
            print(f"Processing order: {order}")
            # get the order id thats running
            # @todo can't use "order_contains_name" because using ROI_77 will return ROI_777 and ROI_77
            success_order_ids = await download.get_order_ids_by_name(
                self.orders_client,
                order.name,
                states=["success"],
                contains_name=order_contains_name,
            )

            running_order_ids = await download.get_order_ids_by_name(
                self.orders_client,
                order.name,
                states=["running"],
                contains_name=order_contains_name,
            )

            print(f"Success order ids: {success_order_ids}")
            print(f"Running order ids: {running_order_ids}")

            # Workflow to create a new order and download it
            # create order requests
            # create list of existing order to download

            # create new orders in parallel with list of order request
            # await new orders to be created
            # while the orders are being created download any existing orders
            # download new orders

            # for success order id get the existing order
            if success_order_ids:
                for order_id in success_order_ids:
                    # this gets a JSON description of the order
                    existing_order = await self.orders_client.get_order(order_id)

                    # read the ROI's id and geometry from the order
                    roi = read_roi_from_order(order)
                    roi_id = roi["roi_id"]
                    roi_geometry = roi["roi_geometry"]

                    # create a destination path with the order name and the destination
                    destination = pathlib.Path(order.destination, order.name)

                    order_to_download = {
                        "order": existing_order,
                        "destination": destination,
                        "roi_id": roi_id,
                        "roi_geometry": roi_geometry,
                    }
                    orders_to_download.append(order_to_download)

                    # @todo check if the order is already downloaded and remobe the files that are already downloaded
                    # if not await self.check_if_order_downloaded(roi_id, order.name):

                # @todo: Note in the future we can use files_to_skip to skip files that are already downloaded

                await asyncio.gather(
                    *[
                        self.download_order(
                            order["order"],
                            order["destination"],
                            order["roi_id"],
                            order["roi_geometry"],
                        )
                        for order in orders_to_download
                    ]
                )
            else:
                # create a new order

                # this creates the new orders and downloads the order @todo remove this later
                # await self.create_new_order(self.planet_session, order)

                # what we actually want is to create the new order, then download it
                order_requests = await self.create_new_order_requests(
                    self.planet_session, order
                )
                # this creates and downloads the new order
                await self.download_new_order(order, order_requests)

            # elif running_order_ids:
            #     order_id = running_order_ids[0]
            #     existing_order = await self.orders_client.get_order(order_id)
            #     roi_geometry = self.extract_geometry(existing_order)
            #     roi_id = self.db.get_roi_id_by_name(order.name)
            #     if roi_geometry is None:
            #         raise ValueError(f"Order {order.name} does not have a valid geometry.")
            #     # check if the order is already downloaded
            #     if not await self.check_if_order_downloaded(roi_id, order.name):
            #         await self.download_order(existing_order, order.destination, roi_id, roi_geometry)
            # else:
            #     # create a new order
            #     new_order = await self.create_order(order.get_order())
            #     roi_geometry = self.extract_geometry(new_order)
            #     roi_id = self.db.get_roi_id_by_name(order.name)
            #     if roi_geometry is None:
            #         raise ValueError(f"Order {order.name} does not have a valid geometry.")
            #     # download the new order
            #     await self.download_order(new_order, order.destination, roi_id, roi_geometry)

    async def download_order(
        self,
        order: str,
        directory: str,
        roi_id: str,
        roi_geometry: dict,
        files_to_skip: list = None,
    ):
        """
        Downloads the order and its items to the specified directory.

        Args:
            order (dict): JSON description of the order recieved from "orders_client.get_order(order_id)".
            directory (str): The directory to download the items to.
            roi_id (str): The region of interest (ROI) identifier.
            roi_geometry (dict): the geometry of the region of interest in GeoJSON format. Must contain "coordinates" as a top level key.
                Example:{
                    "type": "Polygon",
                    "coordinates": [...]
                }
            files_to_skip (list): A list of files to skip downloading.
                If None, all files will be downloaded.

        Returns:
            None

        Raises:
            Any exceptions raised during the download process will propagate to the caller.
        """
        # ASSUME for now that the order is in a success state
        # ASSUME that none of the files are downloaded yet @todo account for this later
        order_id = order["id"]
        all_items = [
            {
                "location": r["location"],
                "directory": directory / Path(r["name"]).parent,
                "filename": Path(r["name"]).name,
            }
            for r in order["_links"].get("results", [])
            if r
        ]

        # remove those items that should not be downloaded from the list
        if files_to_skip:
            all_items = [
                item for item in all_items if item["filename"] not in files_to_skip
            ]

        # Get all the items except the manifest.json file
        items_to_download = [
            item for item in all_items if item["filename"] != "manifest.json"
        ]
        # Get the manifest.json file
        manifest_items = [
            item for item in all_items if item["filename"] == "manifest.json"
        ]

        progress = tqdm_asyncio(
            total=len(items_to_download) + len(manifest_items),
            desc=f"Downloading order {order['name']}",
        )
        # these functions insert the manifest and tile ids into the database
        await self.insert_order_manifest(
            manifest_items, roi_id, order_id, roi_geometry, progress
        )
        await self.insert_roi_ids(items_to_download, roi_id, roi_geometry)

        for item in items_to_download:
            print(f'tile_id: {"_".join(item["filename"].split("_")[:4])}')
            await self.processor.process(
                {
                    "action": "update_metadata_roi",
                    "roi_id": roi_id,
                    "tile_id": "_".join(item["filename"].split("_")[:4]),
                    "order_id": order_id,
                    "filepath": item["directory"] / item["filename"],
                    "status": TILE_STATUSES["PENDING"],
                }
            )
            await self.download_with_retry(
                item, roi_id, order_id, "update_metadata_roi", roi_geometry, progress
            )

        progress.close()

    async def insert_order_manifest(
        self, manifest_items, roi_id, order_id, roi_geometry, progress
    ):
        """
        Asynchronously inserts order's manifest_items (aka manifest.json file) into the processing pipeline and then downloads the manifest items with retry logic.

        Args:
            manifest_items (list): A list of manifest items, where each item is a dictionary containing
                "directory" (pathlib.Path) and "filename" (str) keys.
                Example:
                  [{"directory": Path("path/to/dir"), "filename": "manifest.json"}]

            roi_id (str): The region of interest (ROI) identifier.
            order_id (str): The order identifier created by the Planet API.
            roi_geometry (dict): The geometry of the region of interest, typically in GeoJSON format.
            progress (asyncio.Queue): An asynchronous queue to track the progress of the download process.

        Returns:
            None

        Raises:
            Any exceptions raised during processing or downloading will propagate to the caller.
        """
        for item in manifest_items:
            await self.processor.process(
                {
                    "action": "update_order",
                    "roi_id": roi_id,
                    "order_id": order_id,
                    "filepath": item["directory"] / item["filename"],
                    "status": TILE_STATUSES["PENDING"],
                    "geometry": roi_geometry,
                }
            )
            await self.download_with_retry(
                item, roi_id, order_id, "update_order", roi_geometry, progress
            )

    async def insert_tile_ids(
        self, items: List[Dict[str, Any]], geometry: Dict[str, Any]
    ) -> None:
        """
        Extracts unique tile IDs from filenames and queues them for insertion.

        Example:
            items = [
                {"filename": "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"},
                {"filename": "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"},
            ]
            geometry = {"type": "Polygon", "coordinates": [...]}

        Args:
            items (List[Dict[str, Any]]): List of dicts with a 'filename' key.
            geometry (Dict[str, Any]): ROI geometry as GeoJSON. MUST contain "coordinates" as a top-level key.

        Returns:
            None
        """
        # get the tile id from the filename example: "20241004_223419_50_24b7" from "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"
        unique_ids = set("_".join(item["filename"].split("_")[:4]) for item in items)
        unique_ids.discard("manifest.json")
        for tile_id in unique_ids:
            await self.processor.process(
                {
                    "action": "insert_tile",
                    "tile_id": tile_id,
                    "capture_time": extract_unique_datetime_ids(tile_id),
                    "geometry": geometry,
                }
            )

    async def insert_roi_ids(
        self, items: List[Dict[str, Any]], roi_id: str, roi_geometry: Dict[str, Any]
    ) -> None:
        """
        Extracts unique tile IDs from filenames and queues them for insertion.

        Example:
            items = [
                {"filename": "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"},
                {"filename": "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"},
            ]
            roi_id = "roi123"
            roi_geometry = {"type": "Polygon", "coordinates": [...]}

        Args:
            items (List[Dict[str, Any]]): List of dicts with a 'filename' key.
            roi_id (str): Region of interest ID.
            roi_geometry (Dict[str, Any]): ROI geometry as GeoJSON. MUST contain "coordinates" as a top-level key.

        Returns:
            None
        """
        # get the tile id from the filename example: "20241004_223419_50_24b7" from "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"
        unique_ids = set("_".join(item["filename"].split("_")[:4]) for item in items)
        unique_ids.discard("manifest.json")
        for tile_id in unique_ids:
            await self.processor.process(
                {
                    "action": "insert_roi",
                    "roi_id": roi_id,
                    "tile_id": tile_id,
                    "capture_time": extract_unique_datetime_ids(tile_id),
                    "geometry": roi_geometry,
                }
            )

    async def download_with_retry(
        self,
        item: Dict[str, Any],
        roi_id: str,
        order_id: str,
        action: str,
        geometry: Dict[str, Any],
        progress: tqdm_asyncio,
        max_attempts: int = 3,
    ) -> None:
        """
        Attempts to download an asset with retry logic. Marks the tile as downloaded or failed.

        Args:
            item (Dict[str, Any]): Asset metadata containing 'location', 'filename', and 'directory'.
            roi_id (str): Region of interest ID.
            order_id (str): Planet order ID.
            action (str): Action type for processing ('update_order', etc.).
            geometry (Dict[str, Any]): GeoJSON geometry of the tile.
            progress (tqdm_asyncio): Progress bar for tracking download progress.
            max_attempts (int): Number of retry attempts (default: 3).
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                item["directory"].mkdir(parents=True, exist_ok=True)
                path = item["directory"]

                await self.orders_client.download_asset(
                    item["location"],
                    filename=item["filename"],
                    directory=path,
                    overwrite=True,
                    progress_bar=True,
                )
                await self.processor.process(
                    {
                        "action": action,
                        "roi_id": roi_id,
                        "tile_id": "_".join(item["filename"].split("_")[:4]),
                        "order_id": order_id,
                        "filepath": item["directory"] / item["filename"],
                        "status": TILE_STATUSES["DOWNLOADED"],
                        "geometry": geometry,
                    }
                )
                progress.update(1)
                return
            except asyncio.CancelledError as e:
                print("[CANCELLED] Download was cancelled. Cleaning up.")
                raise e  # re-raise to let it bubble up cleanly
            except Exception:
                attempts += 1
                await asyncio.sleep(1)
        await self.processor.process(
            {
                "action": action,
                "roi_id": roi_id,
                "order_id": order_id,
                "tile_id": "_".join(item["filename"].split("_")[:4]),
                "filepath": item["directory"] / item["filename"],
                "status": TILE_STATUSES["FAILED"],
                "geometry": geometry,
            }
        )
        progress.update(1)

    async def create_new_order_requests(self, session, order: Order):
        """
        Creates a series of new order requests based on the order specified in the order object.
        Note: if the requested data has more than 500 files available its split into smaller suborders of less than 500

        Args:

                session (planet.Session): The planet session to use for the order creation.
                order (Order): The order object containing the order details.

        Returns:
            list: A list of order requests to be created.

        """
        # get the order dictionary from the order object
        order_dict = order.to_dict()
        order_name = order.name
        roi_gdf = order.get_roi_geodataframe()

        # this ROI ID needs to be created in the database before the order
        roi_id = order_dict.get("roi_id", "")
        # if the ROI ID already exists lets check if the geometry is the same as the one in the database
        # If the ROI ID & geometry are the same then we can use the existing ROI ID
        # if the ROI ID is the same but the geometry is different warn the user and STOP the order creation
        if roi_id == "":
            raise ValueError(f"ROI ID not found in order dictionary: {order_dict}")
        # @todo: If the ROI ID is found in the database, we should check if the ROI ID is valid and if it exists in the database.
        # If it does exist add the order name to the ROI_ID then store it in the database
        # self.db.validate_roi_id(roi_id)
        # convert the ROI to a dictionary so it can be used with the Planet API
        roi_dict = json.loads(roi_gdf.to_json())
        start_date, end_date = order.dates
        cloud_cover = order_dict.get("cloud_cover", CLOUD_COVER)
        min_area_percentage = order_dict.get("min_area_percentage", MIN_AREA_PERCENT)
        months_filter = order.month_filter
        product_bundle = order.product_bundle
        item_type = order.item_type

        tools = order.tools

        item_list = await download.search_for_items(
            session, roi_dict, start_date, end_date, cloud_cover=cloud_cover
        )

        # filter the items list by area. If the area of the image is less than than percentage of area of the roi provided, then the image is not included in the list
        print(f"Number of items to download before filtering by area: {len(item_list)}")
        item_list = download.filter_items_by_area(
            roi_gdf, item_list, min_area_percentage
        )
        print(f"Number of items to download after filtering by area: {len(item_list)}")

        ids = download.get_ids(item_list, months_filter)

        coregister_tool = tools.get("coregister", False)
        coregister_id = order_dict.get("coregister_id", "")
        if coregister_tool:
            coregister_id = download.get_coregister_id(item_list, ids, coregister_id)

        # Split the ids list into batches of 499 or less
        id_batches = download.create_batches(ids, id_to_coregister=coregister_id)
        print(f"Total number of batches: {len(id_batches)}")

        tool_list = download.get_tools(
            roi=roi_dict,
            clip=tools.get("clip", True),
            toar=tools.get("toar", True),
            coregister=coregister_tool,
            id_to_coregister=coregister_id,
        )

        requests = []

        for batch_number, id_batch in enumerate(id_batches):
            order_name = f"{order_name}_batch_{batch_number + 1}"

            request = download.create_order_request(
                order_name, id_batch, tool_list, product_bundle, item_type=item_type
            )
            requests.append(request)
        return requests

    async def download_new_order(self, order: Order, requests: list[dict]):
        """
        Creates a series of new orders based on the order request and download them
        Note: if the requested data has more than 500 files available its split into smaller suborders of less than 500

        Args:

                order (Order): The order object containing the order details.
                requests (list): A list of order requests to be created.

        """
        # get the order dictionary from the order object
        order_dict = order.to_dict()

        roi = read_roi_from_order(order)
        roi_dict = roi["roi_geometry"]
        roi_id = roi["roi_id"]

        # CONSTRAINT this ROI ID needs to be created in the database before the order

        # if the ROI ID already exists lets check if the geometry is the same as the one in the database
        # If the ROI ID & geometry are the same then we can use the existing ROI ID
        # if the ROI ID is the same but the geometry is different warn the user and STOP the order creation

        # @todo: If the ROI ID is found in the database, we should check if the ROI ID is valid and if it exists in the database.
        # If it does exist add the order name to the ROI_ID then store it in the database
        # self.db.validate_roi_id(roi_id)

        download_path = order_dict.get("destination", os.getcwd())
        # make the download path the destionation + order name
        download_path = pathlib.Path(download_path) / order.name
        print(f"Download path: {download_path}")

        async with self.order_semaphore:
            # this creates an order and waits for it to be in state "success" (aka downloadable)
            # Note: this is creating coroutines which is why we are not awaiting them
            order_tasks = [self.await_order(request) for request in requests]

            download_tasks = []
            for order_coro in asyncio.as_completed(order_tasks):
                # get the order object from the order that was created in await_order
                planet_order = await order_coro

                # planet order is a dictionary that shows the contents of the order, we need to get the order to download its contents now
                order_id = planet_order.get("id", "")
                # Now the order is in a downloadable state (aka '_links' in the existing order dictionary contains links to download the order contents)
                existing_order = await self.orders_client.get_order(order_id)

                print(f"planet order: {existing_order}")
                # Immediately launch download
                download_tasks.append(
                    asyncio.create_task(
                        self.download_order(
                            existing_order, download_path, roi_id, roi_dict
                        )
                    )
                )

            # Wait for all downloads to complete
            print(f"Waiting for all downloads to complete")
            await asyncio.gather(*download_tasks)

    # @todo I think this function can be removed
    async def create_new_order(self, session, order: Order):
        """
        Creates a series of new orders based on the order request.
        Note: if the requested data has more than 500 files available its split into smaller suborders of less than 500

        Args:

                session (planet.Session): The planet session to use for the order creation.
                order (Order): The order object containing the order details.



        """
        # get the order dictionary from the order object
        order_dict = order.to_dict()
        order_name = order.name
        roi_gdf = order.get_roi_geodataframe()

        # this ROI ID needs to be created in the database before the order
        roi_id = order_dict.get("roi_id", "")
        # if the ROI ID already exists lets check if the geometry is the same as the one in the database
        # If the ROI ID & geometry are the same then we can use the existing ROI ID
        # if the ROI ID is the same but the geometry is different warn the user and STOP the order creation
        if roi_id == "":
            raise ValueError(f"ROI ID not found in order dictionary: {order_dict}")
        # @todo: If the ROI ID is found in the database, we should check if the ROI ID is valid and if it exists in the database.
        # If it does exist add the order name to the ROI_ID then store it in the database
        # self.db.validate_roi_id(roi_id)

        roi_geometry = roi_gdf.geometry.values[0]
        print(f"ROI geometry: {roi_geometry}")  # @todo remove this later

        # convert the ROI to a dictionary so it can be used with the Planet API
        roi_dict = json.loads(roi_gdf.to_json())
        start_date, end_date = order.dates
        cloud_cover = order_dict.get("cloud_cover", CLOUD_COVER)
        min_area_percentage = order_dict.get("min_area_percentage", MIN_AREA_PERCENT)
        months_filter = order.month_filter
        download_path = order_dict.get("destination", os.getcwd())
        product_bundle = order.product_bundle
        item_type = order.item_type

        tools = order.tools

        item_list = await download.search_for_items(
            session, roi_dict, start_date, end_date, cloud_cover=cloud_cover
        )

        # filter the items list by area. If the area of the image is less than than percentage of area of the roi provided, then the image is not included in the list
        print(f"Number of items to download before filtering by area: {len(item_list)}")
        item_list = download.filter_items_by_area(
            roi_gdf, item_list, min_area_percentage
        )
        print(f"Number of items to download after filtering by area: {len(item_list)}")

        ids = download.get_ids(item_list, months_filter)

        coregister_tool = tools.get("coregister", False)
        coregister_id = order_dict.get("coregister_id", "")
        if coregister_tool:
            coregister_id = download.get_coregister_id(item_list, ids, coregister_id)

        # Split the ids list into batches of 499 or less
        id_batches = download.create_batches(ids, id_to_coregister=coregister_id)
        print(f"Total number of batches: {len(id_batches)}")

        tool_list = download.get_tools(
            roi=roi_dict,
            clip=tools.get("clip", True),
            toar=tools.get("toar", True),
            coregister=coregister_tool,
            id_to_coregister=coregister_id,
        )

        requests = []

        for id_batch in id_batches:
            order_name = f"{order_name}_{len(id_batch)}"
            request = download.create_order_request(
                order_name, id_batch, tool_list, product_bundle, item_type=item_type
            )
            requests.append(request)

        async with self.order_semaphore:
            # this creates an order and waits for it to be in state "success" (aka downloadable)
            order_tasks = [self.await_order(request) for request in requests]

            download_tasks = []
            for order_coro in asyncio.as_completed(order_tasks):
                # get the order object from the order that was created in await_order
                planet_order = await order_coro
                # Immediately launch download
                download_tasks.append(
                    asyncio.create_task(
                        self.download_order(
                            planet_order, download_path, roi_id, roi_geometry
                        )
                    )
                )

            # Wait for all downloads to complete
            print(f"gathering all the {len(download_tasks)} download tasks")
            await asyncio.gather(*download_tasks)
