# db/tile_repository.py

import json
from coastseg_planet.db.base import parse_capture_time, BaseDuckDB


class TileRepository:
    """
    Handles operations related to the 'tiles' and 'tile_files' tables.

    Attributes:
        db (BaseDuckDB): Shared database connection.
    """

    def __init__(self, base: BaseDuckDB):
        self.db = base

    def insert_tile(self, tile_id, capture_time, geom=None, order_name=None):
        cursor = self.db.get_cursor()
        capture_time = parse_capture_time(capture_time)
        geom_param = json.dumps(geom) if geom else None

        cursor.execute("SELECT 1 FROM tiles WHERE tile_id = ?", (tile_id,))
        if cursor.fetchone():
            print(f"[INFO] Tile {tile_id} already exists.")
            return

        query = (
            "INSERT INTO tiles (tile_id, geometry, capture_time, order_name) VALUES (?, ST_GeomFromGeoJSON(?), ?, ?)"
            if geom_param
            else "INSERT INTO tiles (tile_id, geometry, capture_time, order_name) VALUES (?, ?, ?, ?)"
        )
        cursor.execute(query, (tile_id, geom_param, capture_time, order_name))
        self.db.commit()

    def update_tile_geometry(self, tile_id, geom):
        cursor = self.db.get_cursor()
        geom_param = json.dumps(geom)

        # Check if the tile_id exists
        cursor.execute("SELECT 1 FROM tiles WHERE tile_id = ?", (tile_id,))
        if not cursor.fetchone():
            raise ValueError(f"Tile with ID {tile_id} does not exist.")

        cursor.execute(
            "UPDATE tiles SET geometry = ST_GeomFromGeoJSON(?) WHERE tile_id = ?",
            (geom_param, tile_id),
        )
        self.db.commit()

    def insert_tile_file(self, filepath, tile_id, filename, status, order_id=None):
        cursor = self.db.get_cursor()
        cursor.execute(
            """
            INSERT INTO tile_files (filepath, tile_id, filename, order_id, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                tile_id = EXCLUDED.tile_id,
                filename = EXCLUDED.filename,
                order_id = EXCLUDED.order_id,
                status = EXCLUDED.status;
        """,
            (filepath, tile_id, filename, order_id, status),
        )
        self.db.commit()

    def filter_existing_tile_ids(self, tile_ids):
        """
        Filters out tile IDs that already exist in the 'tiles' table.

        Args:
            tile_ids (List[str] | Set[str]): List or set of tile IDs to check.

        Returns:
            List[str]: A list of tile IDs that do NOT exist in the database.
        """
        if not tile_ids:
            return []

        cursor = self.db.get_cursor()
        placeholders = ",".join(["?"] * len(tile_ids))
        query = f"SELECT tile_id FROM tiles WHERE tile_id IN ({placeholders})"
        cursor.execute(query, tuple(tile_ids))

        existing_ids = {row[0] for row in cursor.fetchall()}
        return [tile_id for tile_id in tile_ids if tile_id not in existing_ids]
