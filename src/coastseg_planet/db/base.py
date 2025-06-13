# db/base.py

import duckdb
from datetime import datetime


def parse_capture_time(id_str: str) -> str:
    try:
        date_part, time_part, *_ = id_str.split("_")
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        raise ValueError(f"Invalid datetime format in string: {id_str}") from e


# Centralized SQL for table definitions
TABLE_DEFINITIONS = {
    "roi_data": """
        CREATE TABLE IF NOT EXISTS roi_data (
            roi_id TEXT,
            tile_id TEXT,
            capture_time TIMESTAMP,
            geometry GEOMETRY DEFAULT NULL,
            intersection GEOMETRY DEFAULT NULL,
            PRIMARY KEY (roi_id, tile_id)
        );
    """,
    "roi_files": """
        CREATE TABLE IF NOT EXISTS roi_files (
            filepath TEXT PRIMARY KEY,
            roi_id TEXT,
            tile_id TEXT,
            filename TEXT,
            order_id TEXT,
            status TEXT,
            FOREIGN KEY (roi_id, tile_id) REFERENCES roi_data(roi_id, tile_id)
        );
    """,
    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
            filepath TEXT,
            geometry GEOMETRY DEFAULT NULL
        );
    """,
    "tiles": """
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id TEXT PRIMARY KEY,
            geometry GEOMETRY DEFAULT NULL,
            capture_time TIMESTAMP,
            order_name TEXT DEFAULT NULL
        );
    """,
    "tile_files": """
        CREATE TABLE IF NOT EXISTS tile_files (
            filepath TEXT PRIMARY KEY,
            tile_id TEXT,
            filename TEXT,
            order_id TEXT,
            status TEXT,
            FOREIGN KEY (tile_id) REFERENCES tiles(tile_id)
        );
    """,
}

# Indexes for performance
INDEX_DEFINITIONS = [
    "CREATE INDEX IF NOT EXISTS idx_roi_files_tile_id ON roi_files(tile_id);",
    "CREATE INDEX IF NOT EXISTS idx_tile_files_tile_id ON tile_files(tile_id);",
    "CREATE INDEX IF NOT EXISTS idx_roi_data_tile_id ON roi_data(tile_id);",
    "CREATE INDEX IF NOT EXISTS idx_roi_data_roi_id ON roi_data(roi_id);",
]


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

    def drop_all_tables(self):
        cursor = self.get_cursor()
        for table_name in reversed(list(TABLE_DEFINITIONS.keys())):
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        self.commit()
        print("[SUCCESS] All tables dropped.")

    def create_tables(self):
        self.use_spatial_extension()
        cursor = self.get_cursor()

        # read the table definitions from the dictionary
        for table_name, ddl in TABLE_DEFINITIONS.items():
            cursor.execute(ddl)

        for ddl in INDEX_DEFINITIONS:
            cursor.execute(ddl)

        self.commit()
        print("[SUCCESS] Tables created.")
