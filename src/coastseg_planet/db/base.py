# db/base.py

import duckdb
import json
from datetime import datetime


def parse_capture_time(id_str: str) -> str:
    try:
        date_part, time_part, *_ = id_str.split("_")
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Invalid datetime format in string: {id_str}") from e


class BaseDuckDB:
    def __init__(self, db_path="data.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)

    def get_cursor(self):
        return self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def use_spatial_extension(self):
        cursor = self.get_cursor()
        cursor.execute("INSTALL spatial;")
        cursor.execute("LOAD spatial;")

    def create_tables(self):
        self.use_spatial_extension()
        cursor = self.get_cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rois (
                roi_id TEXT PRIMARY KEY,
                tile_id TEXT,
                geometry GEOMETRY DEFAULT NULL,
                capture_time TIMESTAMP
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS roi_files (
                filepath TEXT PRIMARY KEY,
                roi_id TEXT,
                tile_id TEXT,
                filename TEXT,
                order_id TEXT,
                status TEXT,
                FOREIGN KEY (roi_id) REFERENCES rois(roi_id)
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,
                filepath TEXT,
                geometry GEOMETRY DEFAULT NULL
            );
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tiles (
                tile_id TEXT PRIMARY KEY,
                geometry GEOMETRY DEFAULT NULL,
                capture_time TIMESTAMP
            );
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tile_id ON roi_files(tile_id);")

        self.commit()
        print("[SUCCESS] Tables created.")
