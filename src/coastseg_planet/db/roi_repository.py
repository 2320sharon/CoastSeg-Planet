# db/roi_repository.py

import json
from coastseg_planet.db.base import parse_capture_time, BaseDuckDB


class ROIRepository:
    """
    Manages all database operations related to the 'rois', 'roi_tiles' and 'roi_files' tables.

    Attributes:
        db (BaseDuckDB): Shared database connection and utilities.
    """

    def __init__(self, base: BaseDuckDB):
        """
        Initialize with a shared BaseDuckDB instance.
        """
        self.db = base

    def insert_roi(
        self,
        roi_id,
        tile_id,
        capture_time,
        geom=None,
        intersection=None,
        order_name=None,
    ):
        """
        Inserts a new ROI (if not exists) and links it to a tile via roi_tiles.

        Args:
            roi_id (str): ROI identifier.
            tile_id (str): Tile identifier.
            capture_time (str): Time string in 'YYYYMMDD_HHMMSS_XX' format.
            geom (dict, optional): ROI GeoJSON geometry.
            intersection (dict, optional): The intersection GeoJSON geometry with the tile. Note that this should only be used if the tile exists in the tiles table.
            order_name (str, optional): The name of the order associated with the ROI.
        """
        cursor = self.db.get_cursor()

        # Check if ROI already exists
        cursor.execute("SELECT 1 FROM rois WHERE roi_id = ?", (roi_id,))
        roi_exists = cursor.fetchone()

        # Insert ROI only if it does not exist
        if not roi_exists:
            if geom:
                geom_param = json.dumps(geom)
                cursor.execute(
                    "INSERT INTO rois (roi_id, geometry, order_name) VALUES (?, ST_GeomFromGeoJSON(?), ?)",
                    (roi_id, geom_param, order_name),
                )
            else:
                cursor.execute(
                    "INSERT INTO rois (roi_id, geometry, order_name) VALUES (?, ?, ?)",
                    (roi_id, None, order_name),
                )

        if tile_id:
            # Always insert or update roi_tiles using the internal function
            self.insert_roi_tile(
                roi_id=roi_id,
                tile_id=tile_id,
                capture_time=capture_time,
                intersection=intersection,
                fallback_geom=geom,
            )

        self.db.commit()
        print(f"[SUCCESS] ROI {roi_id} inserted (if new) and linked to tile {tile_id}.")

    def create_roi(self, roi_id, geom=None):
        """
        Inserts a new ROI into the 'rois' table (without linking to any tiles).

        Args:
            roi_id (str): Unique identifier for the ROI.
            geom (dict, optional): GeoJSON geometry.
        """
        cursor = self.db.get_cursor()
        cursor.execute("SELECT 1 FROM rois WHERE roi_id = ?", (roi_id,))
        if cursor.fetchone():
            print(f"[INFO] ROI {roi_id} already exists.")
            return

        if geom:
            geom_param = json.dumps(geom)
            cursor.execute(
                "INSERT INTO rois (roi_id, geometry) VALUES (?, ST_GeomFromGeoJSON(?))",
                (roi_id, geom_param),
            )
        else:
            cursor.execute(
                "INSERT INTO rois (roi_id, geometry) VALUES (?, ?)", (roi_id, None)
            )

        self.db.commit()
        print(f"[SUCCESS] ROI {roi_id} inserted.")

    def insert_roi_tile(
        self, roi_id, tile_id, capture_time, intersection=None, fallback_geom=None
    ):
        """
        Inserts or updates a row in the 'roi_tiles' table.

        Args:
            roi_id (str): ROI identifier (must exist in 'rois').
            tile_id (str): Tile identifier (does NOT need to exist in 'tiles').
            capture_time (str): Time string in 'YYYYMMDD_HHMMSS_XX' format.
            intersection (dict, optional): GeoJSON of the intersected area (tile clipped to ROI).
            fallback_geom (dict, optional): If intersection is not provided, this geometry is used instead.
                                            Commonly this is the original ROI geometry.
        """
        cursor = self.db.get_cursor()
        parsed_time = parse_capture_time(capture_time)

        intersection_param = (
            json.dumps(intersection)
            if intersection
            else (json.dumps(fallback_geom) if fallback_geom else None)
        )

        if intersection_param:
            cursor.execute(
                """
                INSERT INTO roi_tiles (roi_id, tile_id, capture_time, intersection)
                VALUES (?, ?, ?, ST_GeomFromGeoJSON(?))
                ON CONFLICT(roi_id, tile_id) DO UPDATE SET
                    capture_time = EXCLUDED.capture_time,
                    intersection = EXCLUDED.intersection;
                """,
                (roi_id, tile_id, parsed_time, intersection_param),
            )
        else:
            cursor.execute(
                """
                INSERT INTO roi_tiles (roi_id, tile_id, capture_time, intersection)
                VALUES (?, ?, ?, NULL)
                ON CONFLICT(roi_id, tile_id) DO UPDATE SET
                    capture_time = EXCLUDED.capture_time;
                """,
                (roi_id, tile_id, parsed_time),
            )

        self.db.commit()
        print(f"[SUCCESS] roi_tiles entry added for ROI {roi_id} and Tile {tile_id}.")

    def update_roi(self, roi_id, geom=None):
        """
        Updates an existing ROI. Only non-None fields will be updated.

        Args:
            roi_id (str): The ROI identifier to update.
            geom (dict, optional): New GeoJSON geometry. If None, geometry is not modified.
        """
        cursor = self.db.get_cursor()

        # Check if ROI exists
        cursor.execute("SELECT 1 FROM rois WHERE roi_id = ?", (roi_id,))
        if not cursor.fetchone():
            print(f"[ERROR] ROI {roi_id} does not exist. Nothing was updated.")
            return

        updates = []
        params = []

        if geom is not None:
            updates.append("geometry = ST_GeomFromGeoJSON(?)")
            params.append(json.dumps(geom))

        if not updates:
            print(f"[INFO] No updates provided for ROI {roi_id}.")
            return

        query = f"UPDATE rois SET {', '.join(updates)} WHERE roi_id = ?"
        params.append(roi_id)
        cursor.execute(query, params)
        self.db.commit()
        print(f"[SUCCESS] ROI {roi_id} updated.")

    def insert_roi_file(
        self, filepath, roi_id, tile_id, filename, status, order_id=None
    ):
        """
        Inserts or updates a row in the 'roi_files' table.
        Args:
            filepath (str): File path of the ROI file.
            roi_id (str): ROI identifier.
            tile_id (str): Tile identifier.
            filename (str): File name of the ROI file.
            status (str): Status of the file.
            order_id (str, optional): Order ID associated with the file.
        """

        cursor = self.db.get_cursor()
        cursor.execute(
            """
            INSERT INTO roi_files (filepath, roi_id, tile_id, filename, order_id, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                roi_id = EXCLUDED.roi_id,
                tile_id = EXCLUDED.tile_id,
                filename = EXCLUDED.filename,
                order_id = EXCLUDED.order_id,
                status = EXCLUDED.status;
        """,
            (filepath, roi_id, tile_id, filename, order_id, status),
        )
        self.db.commit()

    def get_filepaths_by_status(self, status, order_id=None):
        cursor = self.db.get_cursor()
        query_columns = "filepath, tile_id, filename, order_id, status"
        where_clauses, params = [], []

        if isinstance(status, list):
            placeholders = ",".join(["?"] * len(status))
            where_clauses.append(f"status IN ({placeholders})")
            params.extend(status)
        else:
            where_clauses.append("status = ?")
            params.append(status)

        if order_id is None:
            where_clauses.append("order_id IS NULL")
        else:
            where_clauses.append("order_id = ?")
            params.append(order_id)

        where_clause = " AND ".join(where_clauses)
        query = f"SELECT {query_columns} FROM roi_files WHERE {where_clause}"
        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
