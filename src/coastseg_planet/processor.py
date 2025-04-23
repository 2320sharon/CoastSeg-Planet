import os
import logging
import asyncio
from typing import Dict, List, Optional
from coastseg_planet.config import TILE_STATUSES
from coastseg_planet.new_db3 import DuckDBInterface

class TileProcessor:
    """
    Handles processing of tile-related data operations including inserting tile metadata,
    tile files, and orders into the DuckDB database.
    """

    def __init__(self, db: DuckDBInterface):
        """
        Initializes the TileProcessor with a DuckDBInterface instance.

        Args:
            db (DuckDBInterface): An interface for interacting with the DuckDB database.
        """
        self.db = db

    async def process(self, db_entry: Dict) -> None:
        """
        Processes a database entry according to its specified action.

        Args:
            db_entry (Dict): A dictionary containing details about the tile/order action.
        """
        try:
            action = db_entry.get('action', 'update_metadata')
            if action == 'update_metadata':
                logging.info(f"[Processor] Saving {db_entry['roi_id']} with status: {db_entry.get('status', '')}")
                await self.insert_tile_file(db_entry)
            elif action == 'insert_tile':
                logging.info(f"[Processor] Inserting tile {db_entry['roi_id']} into Tiles Table with geometry {db_entry.get('geometry', '')}")
                await self.insert_tile(db_entry)
            elif action == 'update_order':
                logging.info(f"[Processor] Inserting order {db_entry['order_id']} into orders Table with geometry {db_entry.get('geometry', '')}")
                await self.insert_order(db_entry)
            else:
                logging.error(f"[Processor] Unknown action: {action}")
        except Exception as e:
            logging.error(f"[Processor] Error during process: {e}", exc_info=True)
            raise e

    async def insert_tile(self, entry: Dict) -> None:
        """
        Inserts or updates a tile entry in the database.

        Args:
            entry (Dict): Dictionary containing tile data including roi_id, tile_id, geometry, and capture_time.
        """
        roi_id = entry['roi_id']
        tile_id = entry['tile_id']
        geometry = entry['geometry']
        capture_time = entry['capture_time']

        if geometry is None:
            logging.warning(f"[Processor] Geometry is None for {roi_id}")
            return

        self.db.insert_tile(
            roi_id=roi_id,
            tile_id=tile_id,
            capture_time=capture_time,
            geom=geometry,
        )
        logging.info(f"[Processor] Inserted {roi_id} into Tiles Table")

    async def insert_order(self, entry: Dict) -> None:
        """
        Inserts or updates an order entry in the database.

        Args:
            entry (Dict): Dictionary containing order data including order_id, filepath, geometry, and status.
        """
        order_id = entry['order_id']
        filepath = entry['filepath']
        filename = os.path.basename(filepath)
        status = entry.get('status', "unknown")
        geometry = entry.get('geometry')

        self.db.insert_order(
            order_id=order_id,
            filename=filename,
            filepath=filepath,
            geometry=geometry,
            status=status,
        )
        logging.info(f"[Processor] Inserted {order_id} into orders Table")

    async def insert_tile_file(self, entry: Dict) -> None:
        """
        Inserts or updates a tile file entry in the database.

        Args:
            entry (Dict): Dictionary containing tile file data including roi_id, tile_id, filepath, status, and order_id.
        """
        roi_id = entry['roi_id']
        tile_id = entry['tile_id']
        filepath = entry['filepath']
        filename = os.path.basename(filepath)
        status = entry['status']
        order_id = entry['order_id']

        self.db.insert_tile_file(
            filepath=filepath,
            roi_id=roi_id,
            tile_id=tile_id,
            filename=filename,
            order_id=order_id,
            status=status,
        )
        logging.info(f"[Processor] Inserted {roi_id} into tile_files Table")

    async def insert_to_db(self, entry: Dict) -> None:
        """
        Simulates an insert into the database. Placeholder for actual DB operations.

        Args:
            entry (Dict): Dictionary representing a database entry.
        """
        await asyncio.sleep(0.1)
        logging.info(f"[Processor] Inserted {entry['roi_id']} into DB")

    def get_failed_or_pending_items(self, items_to_download: List[Dict], order_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieves items that are either downloading, failed, or pending from the database.

        Args:
            items_to_download (List[Dict]): List of items to check.
            order_id (Optional[str]): Filter items by order ID if provided.

        Returns:
            List[Dict]: List of file entries with specified statuses.
        """
        status = [TILE_STATUSES['DOWNLOADING'], TILE_STATUSES['FAILED'], TILE_STATUSES['PENDING']]
        return self.db.get_filepaths_by_status(status=status, order_id=order_id)

    def get_success_file_items(self, items_to_download: List[Dict], order_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieves items that have been successfully downloaded.

        Args:
            items_to_download (List[Dict]): List of items to check.
            order_id (Optional[str]): Filter items by order ID if provided.

        Returns:
            List[Dict]: List of file entries with download success status.
        """
        status = [TILE_STATUSES['DOWNLOADED']]
        return self.db.get_filepaths_by_status(status=status, order_id=order_id)