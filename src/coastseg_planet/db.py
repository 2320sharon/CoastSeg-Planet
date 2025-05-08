import duckdb
import json
from datetime import datetime


class DuckDBInterface:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)

    def get_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
        return self.conn

    def get_cursor(self):
        return self.conn.cursor()

    def close(self):
        self.conn.close()
        self.conn = None

    def use_spatial_extension(self):
        cursor = self.get_cursor()
        cursor.execute("INSTALL spatial;")
        cursor.execute("LOAD spatial;")

    def create_geometry_index(self):
        """
        Creates a spatial index on the geometry column of the rois table.
        """
        cursor = self.get_cursor()

        # Check if the index already exists by name
        cursor.execute(
            """
            SELECT 1 
            FROM duckdb_indexes 
            WHERE table_name = 'rois' AND index_name = 'idx_rois_geometry'
        """
        )
        if cursor.fetchone():
            print("[INFO] Spatial index 'idx_rois_geometry' already exists.")
            return

        cursor.execute("CREATE INDEX idx_rois_geometry ON rois(geometry);")
        self.get_connection().commit()
        print("[SUCCESS] Created spatial index 'idx_rois_geometry' on rois.geometry.")

    def get_table_names(self):
        cursor = self.get_cursor()
        cursor.execute("SELECT table_name FROM duckdb_tables;")
        tables = cursor.fetchall()
        return [table[0] for table in tables]

    def create_tables(self):
        cursor = self.get_cursor()

        cursor.execute(
            """
            SELECT COUNT(*) FROM duckdb_tables 
            WHERE table_name IN ('rois', 'roi_files', 'orders')
        """
        )
        if cursor.fetchone()[0] == 3:
            print("[INFO] Tables already exist. Skipping creation.")
            return

        self.use_spatial_extension()

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

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tile_id ON roi_files(tile_id);")
        print("[SUCCESS] Tables created.")

    def get_roi_ids(self):
        """
        Returns a set of all ROI_IDs in the rois table.
        """
        cursor = self.get_cursor()
        cursor.execute("SELECT roi_id FROM rois;")
        return set(row[0] for row in cursor.fetchall())

    # get all the ROIs ID in between the start and end date
    def get_roi_ids_by_date(self, start_date, end_date):
        """
        Returns a set of all ROI_IDs in the rois table that fall within the specified date range.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            set: A set of ROI_IDs that fall within the specified date range.
        """
        cursor = self.get_cursor()
        cursor.execute(
            """
            SELECT roi_id FROM rois 
            WHERE capture_time BETWEEN ? AND ?;
        """,
            (start_date, end_date),
        )
        return set(row[0] for row in cursor.fetchall())

    def get_all_tile_ids(self):
        """
        Returns a set of all ROI_IDs in the rois table.
        """
        cursor = self.get_cursor()
        cursor.execute("SELECT tile_id FROM rois;")
        return set(row[0] for row in cursor.fetchall())

    def insert_tile(self, tile_id, capture_time, geom=None):
        cursor = self.get_cursor()
        geom_param = json.dumps(geom) if geom else None

        def format_capture_time(id_str: str) -> str:
            """
            Converts a string in the format 'YYYYMMDD_HHMMSS_XX' to 'YYYY-MM-DD HH:MM:SS'.

            Args:
                id_str (str): The raw datetime ID string (e.g., '20241001_214231_42').

            Returns:
                str: A formatted timestamp string (e.g., '2024-10-01 21:42:31').

            Raises:
                ValueError: If the input does not match the expected format.
            """
            try:
                date_part, time_part, *_ = id_str.split("_")
                dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                raise ValueError(f"Invalid datetime format in string: {id_str}") from e

        capture_time = format_capture_time(capture_time)
        cursor.execute("SELECT 1 FROM tiles WHERE tile_id = ?", (tile_id,))
        if cursor.fetchone():
            print(f"[INFO] Tile with tile_id '{tile_id}' already exists.")
            return

        query = (
            """
            INSERT INTO tiles ( tile_id, geometry,capture_time)
            VALUES ( ?, ST_GeomFromGeoJSON(?), ?)
        """
            if geom_param
            else """
            INSERT INTO tiles (tile_id, geometry, capture_time)
            VALUES (?, ?, ?)
        """
        )

        cursor.execute(query, (tile_id, geom_param, capture_time))
        self.get_connection().commit()
        print(f"[SUCCESS] Inserted tile with tile_id '{tile_id}'.")

    def insert_roi(self, roi_id, tile_id, capture_time, geom=None):
        cursor = self.get_cursor()
        geom_param = json.dumps(geom) if geom else None

        def format_capture_time(id_str: str) -> str:
            """
            Converts a string in the format 'YYYYMMDD_HHMMSS_XX' to 'YYYY-MM-DD HH:MM:SS'.

            Args:
                id_str (str): The raw datetime ID string (e.g., '20241001_214231_42').

            Returns:
                str: A formatted timestamp string (e.g., '2024-10-01 21:42:31').

            Raises:
                ValueError: If the input does not match the expected format.
            """
            try:
                date_part, time_part, *_ = id_str.split("_")
                dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                raise ValueError(f"Invalid datetime format in string: {id_str}") from e

        capture_time = format_capture_time(capture_time)
        cursor.execute("SELECT 1 FROM rois WHERE roi_id = ?", (roi_id,))
        if cursor.fetchone():
            print(f"[INFO] ROI with ID '{roi_id}' already exists.")
            return

        query = (
            """
            INSERT INTO rois (roi_id, tile_id, geometry, capture_time)
            VALUES (?, ?, ST_GeomFromGeoJSON(?), ?)
        """
            if geom_param
            else """
            INSERT INTO rois (roi_id, tile_id, geometry, capture_time)
            VALUES (?, ?, ?, ?)
        """
        )

        cursor.execute(query, (roi_id, tile_id, geom_param, capture_time))
        self.get_connection().commit()
        print(f"[SUCCESS] Inserted ROI with ID '{roi_id}'.")

    def insert_order(
        self, order_id, filename, status=None, filepath=None, geometry=None
    ):
        """
        Inserts or updates an order entry in the orders table.

        Args:
            order_id (str): The unique ID for the order. (Primary key)
            filename (str): The name of the associated file.
            status (str, optional): Status of the order. Example: "pending", "success", etc.
            filepath (str, optional): Path to the associated file.
            geometry (dict, optional): GeoJSON geometry.

        Behavior:
            - If an order with the same order_id exists, updates its values.
            - If not, inserts a new order record.
        """
        print(
            f"[INFO] Inserting or updating order '{order_id}' with file '{filename}'..."
        )

        cursor = self.get_cursor()
        geom_param = json.dumps(geometry) if geometry else None

        try:
            cursor.execute(
                """
                INSERT INTO orders (order_id, filename, status, filepath, geometry)
                VALUES (?, ?, ?, ?, ST_GeomFromGeoJSON(?))
                ON CONFLICT(order_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    status = EXCLUDED.status,
                    filepath = EXCLUDED.filepath,
                    geometry = EXCLUDED.geometry;
                """,
                (order_id, filename, status, filepath, geom_param),
            )
            self.get_connection().commit()
            print(f"[SUCCESS] Inserted or updated order '{order_id}'.")
        except Exception as e:
            print(f"[ERROR] Failed to insert/update order '{order_id}': {e}")
            raise e

    def insert_tile_file(self, filepath, tile_id, filename, status, order_id=None):
        cursor = self.get_cursor()
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
        self.get_connection().commit()
        print(f"[SUCCESS] Inserted or updated tile file '{filepath}'.")

    def insert_roi_file(
        self, filepath, roi_id, tile_id, filename, status, order_id=None
    ):
        cursor = self.get_cursor()
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
        self.get_connection().commit()
        print(f"[SUCCESS] Inserted or updated roi file '{filepath}'.")

    def get_roi_filepaths_by_status(self, status, order_id=None):
        """
        Returns rows from roi_files that have the given status or statuses,
        optionally filtered by order_id.

        Parameters:
            status (str or list of str): One or more status values to filter by.
            order_id (int or None, optional): If provided, filters results to only include
                                            rows with this order_id. If None, includes rows
                                            where order_id IS NULL.

        Returns:
            List[dict]: Rows as dictionaries. Each dictionary contains the following keys:
                - filepath
                - tile_id
                - filename
                - order_id
                - status

        Example Output:
            [
                {
                    'filepath': '/some/path/file1.tif',
                    'tile_id': 'tile_001',
                    'filename': 'file1.tif',
                    'order_id': 42,
                    'status': 'FAILED'
                },
                {
                    'filepath': '/some/path/file2.tif',
                    'tile_id': 'tile_002',
                    'filename': 'file2.tif',
                    'order_id': 42,
                    'status': 'FAILED'
                }
            ]
        """
        cursor = self.get_cursor()
        query_columns = "filepath, tile_id, filename, order_id, status"
        where_clauses = []
        parameters = []

        # Handle status (single or list)
        if isinstance(status, list):
            placeholders = ",".join(["?"] * len(status))
            where_clauses.append(f"status IN ({placeholders})")
            parameters.extend(status)
        else:
            where_clauses.append("status = ?")
            parameters.append(status)

        # Handle optional order_id (including None/null)
        if order_id is None:
            where_clauses.append("order_id IS NULL")
        else:
            where_clauses.append("order_id = ?")
            parameters.append(order_id)

        # Build and run query
        where_clause = " AND ".join(where_clauses)
        query = f"SELECT {query_columns} FROM tile_files WHERE {where_clause}"
        cursor.execute(query, parameters)

        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(column_names, row)) for row in rows]

    def delete_tile(self, tile_id):
        """
        Deletes a tile from the tiles and tile_files tables by tile_ID.
        """
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM tile_files WHERE roi_id = ?", (tile_id,))
        cursor.execute("DELETE FROM tiles WHERE roi_id = ?", (tile_id,))
        self.get_connection().commit()

    def delete_roi(self, roi_id):
        """
        Deletes an roi and all associated files by ROI_ID.
        """
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM roi_files WHERE roi_id = ?", (roi_id,))
        cursor.execute("DELETE FROM rois WHERE roi_id = ?", (roi_id,))
        self.get_connection().commit()

    def delete_tiles(self, tile_ids):
        """
        Deletes multiple tiles and their associated files given a list or set of tile_ids.
        """
        if not tile_ids:
            return

        cursor = self.get_cursor()
        placeholders = ",".join(["?"] * len(tile_ids))
        cursor.execute(
            f"DELETE FROM tile_files WHERE tile_id IN ({placeholders})", tuple(tile_ids)
        )
        cursor.execute(
            f"DELETE FROM tiles WHERE tile_id IN ({placeholders})", tuple(tile_ids)
        )
        self.get_connection().commit()
        print(f"[SUCCESS] Deleted {len(tile_ids)} tiles and associated files.")

    def delete_rois(self, roi_ids):
        """
        Deletes multiple rois and their associated files given a list or set of roi_ids.
        """
        if not roi_ids:
            return

        cursor = self.get_cursor()
        placeholders = ",".join(["?"] * len(roi_ids))
        cursor.execute(
            f"DELETE FROM tile_files WHERE roi_id IN ({placeholders})", tuple(roi_ids)
        )
        cursor.execute(
            f"DELETE FROM tiles WHERE roi_id IN ({placeholders})", tuple(roi_ids)
        )
        self.get_connection().commit()
        print(f"[SUCCESS] Deleted {len(roi_ids)} tiles and associated files.")

    def delete_tile_file(self, filepath):
        """
        Deletes metadata of a tile file.
        """
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM tile_files WHERE filepath = ?", (filepath,))
        self.get_connection().commit()

    def delete_roi_file(self, filepath):
        """
        Deletes metadata of a roi file.
        """
        cursor = self.get_cursor()
        cursor.execute("DELETE FROM roi_files WHERE filepath = ?", (filepath,))
        self.get_connection().commit()

    def get_tile_ids_by_geom(self, roi_geojson):
        """
        Returns tile_ids whose geometries intersect with the given geometry.
        """
        self.use_spatial_extension()
        roi_json_str = json.dumps(roi_geojson)
        cursor = self.get_cursor()
        cursor.execute(
            """
            SELECT tile_id
            FROM tiles
            WHERE ST_Intersects(geometry, ST_GeomFromGeoJSON(?))
            """,
            (roi_json_str,),
        )
        results = cursor.fetchall()
        return [row[0] for row in results]

    def get_roi_ids_by_roi(self, roi_geojson):
        """
        Returns roi_ids whose geometries intersect with the given ROI.

        Parameters:
            roi_geojson (dict): A GeoJSON-like dictionary representing the ROI geometry.

        Returns:
            List[str]: A list of roi_ids whose geometries intersect with the given ROI.

        Example:
            roi_geojson = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.5, 37.7],
                        [-122.4, 37.7],
                        [-122.4, 37.8],
                        [-122.5, 37.8],
                        [-122.5, 37.7]
                    ]
                ]
            }
            roi_ids = db.get_roi_ids_by_roi(roi_geojson)
            print(roi_ids)
            # Output: ['roi_001', 'roi_002', ...]

        """
        self.use_spatial_extension()
        roi_json_str = json.dumps(roi_geojson)
        cursor = self.get_cursor()
        cursor.execute(
            """
            SELECT roi_id
            FROM rois
            WHERE ST_Intersects(geometry, ST_GeomFromGeoJSON(?))
            """,
            (roi_json_str,),
        )
        results = cursor.fetchall()
        return [row[0] for row in results]

    def update_roi_status_by_filepath(self, filepath, status):
        cursor = self.get_cursor()
        cursor.execute(
            """
            UPDATE roi_files SET status = ? WHERE filepath = ?
            """,
            (status, filepath),
        )
        self.get_connection().commit()

    def update_roi_status_by_roi_id(self, roi_id, status):
        cursor = self.get_cursor()
        cursor.execute(
            """
            UPDATE roi_files SET status = ? WHERE roi_id = ?
            """,
            (status, roi_id),
        )
        self.get_connection().commit()

    def query(self, sql, params=None):
        cursor = self.get_cursor()
        cursor.execute(sql, params or ())
        return cursor.fetchall()

    def get_roi_status_by_roi_id(self, roi_id):
        cursor = self.get_cursor()
        cursor.execute(
            """
            SELECT status FROM roi_files WHERE roi_id = ? LIMIT 1
            """,
            (roi_id,),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def get_roi_status_by_filepath(self, filepath):
        cursor = self.get_cursor()
        cursor.execute(
            """
            SELECT status FROM roi_files WHERE filepath = ?
            """,
            (filepath,),
        )
        result = cursor.fetchone()
        return result[0] if result else None
