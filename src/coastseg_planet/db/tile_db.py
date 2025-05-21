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

    def insert_tile(self, tile_id, capture_time, geom=None):
        cursor = self.db.get_cursor()
        capture_time = parse_capture_time(capture_time)
        geom_param = json.dumps(geom) if geom else None

        cursor.execute("SELECT 1 FROM tiles WHERE tile_id = ?", (tile_id,))
        if cursor.fetchone():
            print(f"[INFO] Tile {tile_id} already exists.")
            return

        query = (
            "INSERT INTO tiles (tile_id, geometry, capture_time) VALUES (?, ST_GeomFromGeoJSON(?), ?)"
            if geom_param
            else "INSERT INTO tiles (tile_id, geometry, capture_time) VALUES (?, ?, ?)"
        )
        cursor.execute(query, (tile_id, geom_param, capture_time))
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
