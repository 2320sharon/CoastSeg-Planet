# db/roi_repository.py

import json
from coastseg_planet.db.base import parse_capture_time, BaseDuckDB


class ROIRepository:
    """
    Manages all database operations related to the 'rois' and 'roi_files' tables.

    Attributes:
        db (BaseDuckDB): Shared database connection and utilities.
    """

    def __init__(self, base: BaseDuckDB):
        """
        Initialize with a shared BaseDuckDB instance.
        """
        self.db = base

    def insert_roi(self, roi_id, tile_id, capture_time, geom=None):
        """
        Inserts a new ROI record into the 'rois' table if it doesn't already exist.

        Args:
            roi_id (str): Unique identifier for the ROI.
            tile_id (str): Related tile ID.
            capture_time (str): Capture time in 'YYYYMMDD_HHMMSS_XX' format.
            geom (dict, optional): GeoJSON geometry.
        """
        cursor = self.db.get_cursor()
        geom_param = json.dumps(geom) if geom else None
        capture_time = parse_capture_time(capture_time)

        cursor.execute("SELECT 1 FROM rois WHERE roi_id = ?", (roi_id,))
        if cursor.fetchone():
            print(f"[INFO] ROI {roi_id} already exists.")
            return

        query = (
            "INSERT INTO rois (roi_id, tile_id, geometry, capture_time) VALUES (?, ?, ST_GeomFromGeoJSON(?), ?)"
            if geom_param
            else "INSERT INTO rois (roi_id, tile_id, geometry, capture_time) VALUES (?, ?, ?, ?)"
        )
        cursor.execute(query, (roi_id, tile_id, geom_param, capture_time))
        self.db.commit()

    def insert_roi_file(
        self, filepath, roi_id, tile_id, filename, status, order_id=None
    ):
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
