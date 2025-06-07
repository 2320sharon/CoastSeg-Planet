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
from coastseg_planet.db.base import BaseDuckDB

from coastseg_planet.processor import TileProcessor


def separate_manifest_items(items: List[Dict]) -> (List[Dict], List[Dict]):
    """
    Separates a list of item dictionaries into two lists: one containing all items except those with filename 'manifest.json',
    and another containing only items with filename 'manifest.json'.

    Args:
        items (List[Dict]): A list of dictionaries, each representing an item with a 'filename' key.

    Returns:
        Tuple[List[Dict], List[Dict]]:
            - The first list contains items where 'filename' is not 'manifest.json'.
            - The second list contains items where 'filename' is 'manifest.json'.
    """
    items_to_download = [item for item in items if item["filename"] != "manifest.json"]
    manifest_items = [item for item in items if item["filename"] == "manifest.json"]
    return items_to_download, manifest_items


def get_all_items(order: Dict, directory: str) -> List[Dict]:
    """
    Returns a list of items to be downloaded from the order dictionary.
    This method extracts the download links and item metadata from the order dictionary

    Example:
        order = {
            "_links": {
                "results": [
                    {"location": "https://example.com/file1.tif", "name": "file1.tif"},
                    {"location": "https://example.com/file2.tif", "name": "file2.tif"},
                ]
            }
        }
        directory = "/path/to/download"
    Example output:
        [
            {
                "location": "https://example.com/file1.tif",
                "directory": "/path/to/download/file1",
                "filename": "file1.tif"
            },
            {
                "location": "https://example.com/file2.tif",
                "directory": "/path/to/download/file2",
                "filename": "file2.tif"
            }
        ]

    Args:
        order (Dict): The order dictionary containing download links and item metadata.
        directory (str): The base directory where items will be stored.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - 'location': The download URL or location of the item.
            - 'directory': The target directory as a Path object where the item should be saved.
            - 'filename': The name of the file to be saved.
    """
    results = order.get("_links", {}).get("results", [])
    return [
        {
            "location": result["location"],
            "directory": Path(directory) / Path(result["name"]).parent,
            "filename": Path(result["name"]).name,
        }
        for result in results
        if result
    ]


def filter_skipped_filenames(
    items: List[Dict], files_to_skip: Optional[List[str]]
) -> List[Dict]:
    """
    Filters a list of item dictionaries by excluding those whose "filename" is present in the files_to_skip list.

    Args:
        items (List[Dict]): A list of dictionaries, each representing an item with a "filename" key.
        files_to_skip (Optional[List[str]]): A list of filenames to exclude from the returned list. If None or empty, all items are returned.

    Returns:
        List[Dict]: A filtered list of item dictionaries, excluding those with filenames in files_to_skip.
    """
    if not files_to_skip:
        return items
    return [item for item in items if item["filename"] not in files_to_skip]


def read_geometry_from_metadata(file: str) -> dict:
    """
    Reads the geometry from a PlanetLabs metadata JSON file and returns it as a dictionary.
    Note: Expects the file to contains a "geometry" key at the top level.

    Args:
        file (str): The path to the metadata JSON file.

    Returns:
        dict: The geometry extracted from the metadata file.
        Example: {'coordinates': [[[], ... []]], 'type': 'Polygon'}

    """
    with open(file, "r") as f:
        metadata = json.load(f)
    return metadata["geometry"]


def read_geometry_from_file(filepath: str) -> dict:
    """
    Reads geometry information from a file if it is a metadata JSON file otherwise returns {}.
    Args:
        filepath (str): The path to the file to read.

    Returns:
        dict: The geometry information extracted from the metadata file, or None if the file is not a metadata JSON.
    """
    filename = os.path.basename(filepath)
    if "metadata.json" in filename:
        return read_geometry_from_metadata(filepath)
    return {}


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
        Example: {
            "roi_id": "20240701_222913_34_24ed",
            "roi_geometry": {
                "type": "Polygon",
                "coordinates": [[[], ... []]]
            }
        }
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
        db: BaseDuckDB,
        planet_session: planet.Session,
        client: Optional[OrdersClient] = None,
        order_semaphore: asyncio.Semaphore = asyncio.Semaphore(5),
        download_semaphore: asyncio.Semaphore = asyncio.Semaphore(10),
    ):
        """
        Initializes the DownloadManager with the given parameters.

        Args:
            processor (TileProcessor): The processor to handle tile processing, this handles queuing and processing of the tiles before entering the database.
            db (BaseDuckDB): The database interface for storing metadata about the downloaded tiles.
            planet_session (planet.Session): The Planet session to use for API requests. This is shared across all clients.
            client (planet.Client): The Planet Orders client to use for API requests to the orders API.
            order_semaphore (asyncio.Semaphore): Semaphore for limiting concurrent order creation.
                API requests are limited to 5 requests per second for the activation API as of 5/1/2025
            download_semaphore (asyncio.Semaphore): Semaphore for limiting concurrent downloads.
                API requests are limited to 5 requests per second for the download API as of 5/1/2025


        """
        self.planet_session = planet_session

        self._orders_client = client  # This will be initialized later if not provided
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

    async def process_order(self, order: Order, contains_name: bool):
        success_ids = await self.fetch_order_ids(order.name, ["success"], contains_name)
        running_ids = await self.fetch_order_ids(order.name, ["running"], contains_name)

        print(f"Success order ids: {success_ids}")
        print(f"Running order ids: {running_ids}")

        if success_ids or running_ids:
            await self.download_existing_orders(order, success_ids + running_ids)
        else:
            await self.create_and_download_order(order)

    async def fetch_order_ids(self, name: str, states: List[str], contains_name: bool):
        return await download.get_order_ids_by_name(
            self.orders_client,
            name,
            states=states,
            contains_name=contains_name,
        )

    async def download_existing_orders(self, order: Order, order_ids: List[str]):
        orders_to_download = []

        for order_id in order_ids:
            existing_order = await self.orders_client.get_order(order_id)
            tile_mode = not existing_order.get("tools", {}).get("clip", False)

            print(f"Clip tool is {'disabled' if tile_mode else 'enabled'}")
            print(f"Tile mode is {'enabled' if tile_mode else 'disabled'}")

            roi = read_roi_from_order(order)
            destination = pathlib.Path(order.destination, order.name)

            orders_to_download.append(
                {
                    "order": existing_order,
                    "destination": destination,
                    "roi_id": roi["roi_id"],
                    "roi_geometry": roi["roi_geometry"],
                    "tile_mode": tile_mode,
                }
            )

        # @todo check if the order is already downloaded and remove the files that are already downloaded
        # if not await self.check_if_order_downloaded(roi_id, order.name):

        # @todo: Note in the future we can use files_to_skip to skip files that are already downloaded

        await asyncio.gather(
            *[
                self.download_order(
                    item["order"],
                    item["destination"],
                    item["roi_id"],
                    item["roi_geometry"],
                    tile_mode=item["tile_mode"],
                )
                for item in orders_to_download
            ]
        )

    async def create_and_download_order(self, order: Order):
        order_requests = await self.create_new_order_requests(
            self.planet_session, order
        )
        await self.download_new_order(order, order_requests)

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

        for order in orders:
            print(f"Processing order: {order}")
            # get the order id thats running
            # @todo can't use "order_contains_name" because using ROI_77 will return ROI_777 and ROI_77
            await self.process_order(order, order_contains_name)

    def _select_download_method(
        self,
        item: Dict,
        order_id: str,
        roi_id: str,
        roi_geometry: Dict,
        progress_bar,
        tile_mode: bool,
    ) -> asyncio.Task:
        if tile_mode:
            return self.download_tile(
                item, order_id, roi_geometry, progress_bar, roi_id
            )
        else:
            return self.download_ROI(item, roi_id, order_id, roi_geometry, progress_bar)

    def prepare_download_tasks(
        self,
        items: List[Dict],
        order_id: str,
        roi_id: str,
        roi_geometry: Dict,
        progress_bar,
        tile_mode: bool,
    ) -> List[asyncio.Task]:
        return [
            self._select_download_method(
                item, order_id, roi_id, roi_geometry, progress_bar, tile_mode
            )
            for item in items
        ]

    async def download_order(
        self,
        order: str,
        directory: str,
        roi_id: str,
        roi_geometry: dict,
        files_to_skip: list = None,
        tile_mode: bool = False,
    ):
        """
        Downloads the order and its items to the specified directory.

        Args:
            order (dict): JSON description of the order recieved from "orders_client.get_order(order_id)".
                This should contain the order ID, name, and other metadata.
                Must contain the keys "id", "name", and "_links" with "results" containing the download links.

            directory (str): The directory to download the items to.
            roi_id (str): The region of interest (ROI) identifier.
            roi_geometry (dict): the geometry of the region of interest in GeoJSON format. Must contain "coordinates" as a top level key.
                Example:{
                    "type": "Polygon",
                    "coordinates": [...]
                }
            files_to_skip (list): A list of files to skip downloading.
                If None, all files will be downloaded.
            tile_mode (bool): If True, this means that entire tiles are being downloaded instead of clipped ROIs.

        Returns:
            None

        Raises:
            Any exceptions raised during the download process will propagate to the caller.
        """
        # @todo: replicate the planet api download order logic to wait until the order is in a success state
        # ASSUME for now that the order is in a success state
        # ASSUME that none of the files are downloaded yet @todo account for this later
        order_id = order["id"]
        all_items = get_all_items(order, directory)
        # remove those items that should not be downloaded from the list
        all_items = filter_skipped_filenames(all_items, files_to_skip)

        # Get all the items except the manifest.json file
        items_to_download, manifest_items = separate_manifest_items(all_items)

        progress_bar = tqdm_asyncio(
            total=len(items_to_download) + len(manifest_items),
            desc=f"Downloading order {order['name']}",
        )

        # these functions insert the manifest and tile ids into the database
        await self.insert_order_manifest(
            manifest_items, roi_id, order_id, roi_geometry, progress_bar
        )
        # Insert the TILE or ROI into the database
        if tile_mode:
            await self.insert_tile_ids(items_to_download, roi_geometry)
        else:
            await self.insert_roi_ids(items_to_download, roi_id, roi_geometry)

        # @todo make this instead create a list of asyncio tasks and then await them all at once
        tasks = self.prepare_download_tasks(
            items_to_download, order_id, roi_id, roi_geometry, progress_bar, tile_mode
        )

        # Start the download tasks with a semaphore to limit concurrency
        async with self.download_semaphore:
            await asyncio.gather(*tasks)
        progress_bar.close()

    async def insert_order_manifest(
        self, manifest_items, roi_id, order_id, roi_geometry, progress_bar
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
            progress_bar (tqdm_asyncio): A tqdm progress bar instance for tracking download progress.

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
                item, roi_id, order_id, "update_order", roi_geometry, progress_bar
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

    async def download_tile(
        self,
        item: dict,
        order_id: str,
        roi_geometry: dict,
        progress_bar: tqdm_asyncio,
        roi_id: str = "",
    ):
        """
        Asynchronously downloads a tile, updates its metadata, and processes its geometry.
        This method performs the following steps:
        1. Updates the tile's metadata status to "PENDING".
        2. Attempts to download the tile with retry logic.
        3. After download, reads the geometry from the downloaded file (if available) and updates the tile's geometry.
        Args:
            item (dict): Dictionary containing information about the tile to download, including 'filename' and 'directory'.
            order_id (str): Identifier for the order associated with the tile.
            roi_geometry (dict): The region of interest geometry used for the download.
            progress_bar (tqdm_asyncio): Progress bar instance to update download progress.
            roi_id (str, optional): Identifier for the region of interest. Defaults to "".
        Returns:
            None
        """
        await self.processor.process(
            {
                "action": "update_metadata_tile",
                "tile_id": "_".join(item["filename"].split("_")[:4]),
                "order_id": order_id,
                "filepath": item["directory"] / item["filename"],
                "status": TILE_STATUSES["PENDING"],
            }
        )
        await self.download_with_retry(
            item,
            roi_id,
            order_id,
            "update_metadata_tile",
            roi_geometry,
            progress_bar,
        )

        # read the geometry out of metadata.json and update the tile to have the exact geometry
        # after the tile is downloaded lets check if it was a metadata.json then read the geometry from it
        geometry = read_geometry_from_file(item["directory"] / item["filename"])
        if geometry:
            self.update_tile_ids(
                item,
                geometry=geometry,
            )

    async def download_ROI(
        self,
        item: dict,
        roi_id: str,
        order_id: str,
        roi_geometry: dict,
        progress_bar: tqdm_asyncio,
    ):
        """
        Downloads the ROI item and updates the database with its status.

        Args:
            item (dict): The item to download.
            roi_id (str): The ID of the region of interest.
            order_id (str): The ID of the order.
            roi_geometry (dict): The geometry of the region of interest.
            progress_bar (tqdm_asyncio): The progress bar to update.
        """
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
            item, roi_id, order_id, "update_metadata_roi", roi_geometry, progress_bar
        )

    async def update_tile_ids(
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
        if not isinstance(items, list):
            items = [items]

        # get the tile id from the filename example: "20241004_223419_50_24b7" from "20241004_223419_50_24b7_3B_AnalyticMS_metadata_clip.xml"
        unique_ids = set("_".join(item["filename"].split("_")[:4]) for item in items)
        unique_ids.discard("manifest.json")
        for tile_id in unique_ids:
            await self.processor.process(
                {
                    "action": "update_tile",
                    "tile_id": tile_id,
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
        # @todo: I'm pretty sure that we only need to insert the roi id once
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
        progress_bar: tqdm_asyncio,
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
            progress_bar (tqdm_asyncio): Progress bar for tracking download progress.
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
                progress_bar.update(1)
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
        progress_bar.update(1)

    async def filter_items(
        self,
        session,
        roi_dict,
        start_date,
        end_date,
        cloud_cover,
        min_area_percentage,
        roi_gdf,
        months_filter,
        tools,
    ):
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

        # if the clip tool is not being used filter the ids to download by making sure they don't already exist in the tiles table
        if not tools.get("clip", False):
            print(f"requested ids: {ids}")
            # this is the ids that are NOT in the database
            ids = self.processor.remove_existing_tile_ids(ids)
            # filter items list so that it only contains these ids
            item_list = [item for item in item_list if item["id"] in ids]
            print(
                f"Number of items to download after filtering by existing tile ids: {len(item_list)}"
            )
            print(f"item_list: {item_list}")
            print(f"ids after filtering: {ids}")
            if not ids:
                print(
                    f"No items to download after filtering by existing tile ids. Exiting."
                )
                raise ValueError(
                    "No items to download after filtering by existing tile ids. Exiting."
                )
        return ids, item_list

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

        ids, item_list = await self.filter_items(
            session,
            roi_dict,
            start_date,
            end_date,
            cloud_cover,
            min_area_percentage,
            roi_gdf,
            months_filter,
            tools,
        )

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
                clip_tool = existing_order.get("tools", {}).get("clip", False)
                tile_mode = (
                    not clip_tool
                )  # if the clip tool is not used then we are in tile mode

                print(f"planet order: {existing_order}")
                # Immediately launch download
                download_tasks.append(
                    asyncio.create_task(
                        self.download_order(
                            existing_order, download_path, roi_id, roi_dict, tile_mode
                        )
                    )
                )

            # Wait for all downloads to complete
            print(f"Waiting for all downloads to complete")
            await asyncio.gather(*download_tasks)
