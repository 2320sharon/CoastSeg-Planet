import os
import logging
from typing import Dict, List, Optional
from coastseg_planet.config import TILE_STATUSES

from coastseg_planet.config import TILE_STATUSES
from coastseg_planet.db.roi_repository import ROIRepository
from coastseg_planet.db.tile_repository import TileRepository
from coastseg_planet.db.order_repository import OrderRepository


class TileProcessor:
    """
    Handles processing of tile-related data operations including inserting tile metadata,
    tile files, and orders into the DuckDB database.
    """

    def __init__(
        self,
        roi_repo: ROIRepository,
        tile_repo: TileRepository,
        order_repo: OrderRepository,
    ):
        """
        Initializes the TileProcessor with separate repositories for different domain objects.

        Args:
            roi_repo (ROIRepository): Repository handling ROI-related DB operations.
            tile_repo (TileRepository): Repository handling tile-related DB operations.
            order_repo (OrderRepository): Repository handling order-related DB operations.
        """
        self.roi_repo = roi_repo
        self.tile_repo = tile_repo
        self.order_repo = order_repo

    async def process(self, db_entry: Dict) -> None:
        """
        Processes a database entry according to its specified action.

        Args:
            db_entry (Dict): A dictionary containing details about the tile/order action.
        """
        try:
            action = db_entry.get("action", "update_metadata")
            print(f"[Processor] Processing action: {action}")
            print(f"[Processor] db_entry: {db_entry}")
            if action == "update_metadata_roi":
                logging.info(
                    "[Processor] Saving ROI %s with status: %s",
                    db_entry["roi_id"],
                    db_entry.get("status", ""),
                )
                await self.insert_roi_file(db_entry)
            elif action == "update_metadata_tile":
                logging.info(
                    "[Processor] Saving tile %s with status: %s",
                    db_entry["tile_id"],
                    db_entry.get("status", ""),
                )
                await self.insert_tile_file(db_entry)
            elif action == "insert_tile":
                logging.info(
                    "[Processor] Inserting tile %s into Tiles Table with geometry %s",
                    db_entry["tile_id"],
                    db_entry.get("geometry", ""),
                )
                await self.insert_tile(db_entry)
            elif action == "insert_roi_tile":
                logging.info(
                    "[Processor] Inserting roi %s with tile_id %s into ROI_TILES Table with geometry %s",
                    db_entry["roi_id"],
                    db_entry["tile_id"],
                    db_entry.get("geometry", ""),
                )
                await self.insert_roi_tile(db_entry)
            elif action == "update_tile":
                logging.info(
                    "[Processor] Updating tile %s in Tiles Table with geometry %s",
                    db_entry["tile_id"],
                    db_entry.get("geometry", ""),
                )
                await self.update_tile_geometry(db_entry["tile_id"])
            elif action == "insert_roi":
                logging.info(
                    "[Processor] Inserting roi %s into Rois Table with geometry %s",
                    db_entry["roi_id"],
                    db_entry.get("geometry", ""),
                )
                await self.insert_roi(db_entry)
            elif action == "update_order":
                logging.info(
                    "[Processor] Inserting order %s into orders Table with geometry %s",
                    db_entry["order_id"],
                    db_entry.get("geometry", ""),
                )
                await self.insert_order(db_entry)
            else:
                logging.error("[Processor] Unknown action: %s", action)
        except Exception as e:
            logging.error("[Processor] Error during process: %s", e, exc_info=True)
            raise e

    async def insert_roi(self, entry: Dict) -> None:
        """
        Inserts a region of interest (ROI) entry into the database.

        Args:
            entry (Dict): Must include roi_id, tile_id, capture_time, geometry.
        """
        if entry.get("geometry") is None:
            logging.warning(f"[Processor] Geometry is None for ROI {entry['roi_id']}")
            return

        self.roi_repo.insert_roi(
            roi_id=entry["roi_id"],
            tile_id=entry["tile_id"],
            capture_time=entry["capture_time"],
            geom=entry["geometry"],
            order_name=entry.get("order_name"),
        )
        logging.info(f"[Processor] Inserted ROI {entry['roi_id']}")

    async def insert_roi_tile(self, entry: Dict) -> None:
        """
        Inserts a region of interest (ROI) entry into the database.
        Assumes that the ROI already exists in the ROIs table.

        Args:
            entry (Dict): Must include roi_id, tile_id, capture_time, geometry.
        """
        self.roi_repo.insert_roi_tile(
            roi_id=entry["roi_id"],
            tile_id=entry["tile_id"],
            capture_time=entry["capture_time"],
            intersection=entry.get("intersection"),
            fallback_geom=entry.get("geometry"),
        )
        logging.info(f"[Processor] Inserted ROI {entry['roi_id']}")

    async def insert_tile(self, entry: Dict) -> None:
        """
        Inserts a tile entry into the database.

        Args:
            entry (Dict): Must include tile_id, capture_time, geometry and optionally order_name.
        """
        if entry.get("geometry") is None:
            logging.warning(f"[Processor] Geometry is None for tile {entry['tile_id']}")
            return

        self.tile_repo.insert_tile(
            tile_id=entry["tile_id"],
            capture_time=entry["capture_time"],
            geom=entry["geometry"],
            order_name=entry.get("order_name"),
        )
        logging.info(f"[Processor] Inserted tile {entry['tile_id']}")

    async def update_tile_geometry(self, entry: Dict) -> None:
        """
        Update a tile entry in the database.

        Args:
            entry (Dict): Must include tile_id, capture_time, geometry.
        """
        if entry.get("geometry") is None:
            logging.warning(f"[Processor] Geometry is None for tile {entry['tile_id']}")
            return

        self.tile_repo.update_tile_geometry(
            tile_id=entry["tile_id"],
            geom=entry["geometry"],
        )
        logging.info(f"[Processor] Updated tile {entry['tile_id']}")

    async def insert_order(self, entry: Dict) -> None:
        """
        Inserts or updates an order entry into the database.

        Args:
            entry (Dict): Must include order_id, filepath, and optionally geometry and status.
        """
        filepath = str(entry["filepath"])
        filename = os.path.basename(filepath)

        self.order_repo.insert_order(
            order_id=entry["order_id"],
            filename=filename,
            filepath=filepath,
            geometry=entry.get("geometry"),
            status=entry.get("status", "unknown"),
        )
        logging.info(f"[Processor] Inserted order {entry['order_id']}")

    async def insert_roi_file(self, entry: Dict) -> None:
        """
        Inserts or updates metadata for an ROI file.

        Args:
            entry (Dict): Must include filepath, roi_id, tile_id, order_id, and status.
        """
        filepath = str(entry["filepath"])
        filename = os.path.basename(filepath)

        self.roi_repo.insert_roi_file(
            filepath=filepath,
            roi_id=entry["roi_id"],
            tile_id=entry["tile_id"],
            filename=filename,
            order_id=entry["order_id"],
            status=entry["status"],
        )
        logging.info(f"[Processor] Inserted ROI file for {entry['roi_id']}")

    async def insert_tile_file(self, entry: Dict) -> None:
        """
        Inserts or updates metadata for a tile file.

        Args:
            entry (Dict): Must include filepath, tile_id, order_id, and status.
        """
        filepath = str(entry["filepath"])
        filename = os.path.basename(filepath)

        self.tile_repo.insert_tile_file(
            filepath=filepath,
            tile_id=entry["tile_id"],
            filename=filename,
            order_id=entry["order_id"],
            status=entry["status"],
        )
        logging.info(f"[Processor] Inserted tile file for {entry['tile_id']}")

    def get_failed_or_pending_items(
        self, items_to_download: List[Dict], order_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetches all ROI files that are marked as DOWNLOADING, FAILED, or PENDING.

        Args:
            items_to_download (List[Dict]): (Unused, kept for API compatibility).
            order_id (Optional[str]): Optional filter to narrow results by order.

        Returns:
            List[Dict]: Matching ROI file records.
        """
        # @todo finish this function to use items_to_download or remove parameter
        return self.roi_repo.get_filepaths_by_status(
            status=[
                TILE_STATUSES["DOWNLOADING"],
                TILE_STATUSES["FAILED"],
                TILE_STATUSES["PENDING"],
            ],
            order_id=order_id,
        )

    def get_success_file_items(
        self, items_to_download: List[Dict], order_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetches all ROI files that are marked as DOWNLOADED.

        Args:
            items_to_download (List[Dict]): (Unused, kept for API compatibility).
            order_id (Optional[str]): Optional filter to narrow results by order.

        Returns:
            List[Dict]: Matching ROI file records with successful download status.
        """
        # @todo finish this function to use items_to_download or remove parameter
        return self.roi_repo.get_filepaths_by_status(
            status=[TILE_STATUSES["DOWNLOADED"]],
            order_id=order_id,
        )

    def remove_existing_tile_ids(self, tile_ids: List[str]):
        """
        Filters out tile IDs that already exist in the 'tiles' table.

        Args:
            tile_ids (List[str] | Set[str]): List or set of tile IDs to check.

        Returns:
            List[str]: A list of tile IDs that do NOT exist in the database.
        """
        return self.tile_repo.remove_existing_tile_ids(tile_ids)
