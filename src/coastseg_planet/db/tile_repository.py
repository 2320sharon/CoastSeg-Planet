# db/tile_repository.py

import json
from coastseg_planet.db.base import parse_capture_time, BaseDuckDB
from shapely import wkb
import geopandas as gpd


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

    def get_tile_files_by_tile_id(self, tile_id):
        """
        Retrieves all tile files associated with a specific tile ID.

        Args:
            tile_id (str): The ID of the tile to query.

        Returns:
            list: A list of dictionaries representing the tile files.

        Example:
            [
                {
                    "filepath": "/path/to/tile1.tif",
                    "tile_id": "20250303_190235_45_2511",
                    "filename": "tile1.tif",
                    "order_id": "order1",
                    "status": "downloaded"
                },
                {
                    "filepath": "/path/to/tile2.tif",
                    "tile_id": "20250303_190235_45_2512",
                    "filename": "tile2.tif",
                    "order_id": "order1",
                    "status": "downloaded"
                }
            ]

        """
        cursor = self.db.get_cursor()
        df = cursor.execute(
            "SELECT * FROM tile_files WHERE tile_id = ?", (tile_id,)
        ).fetchdf()
        return df.to_dict(orient="records") if not df.empty else []

    def update_tile_file_location(
        self,
        old_filepath: str,
        new_filepath: str,
    ):
        """
        Updates the file path of a tile file in the database.
        This method changes the file path of a tile file from `old_filepath` to `new_filepath`
        and sets its status to 'downloaded' in the `tile_files` table.
        Args:
            old_filepath (str): The current file path of the tile file in the database.
            new_filepath (str): The new file path to update in the database.
        Returns:
            None
        """
        self.db.get_cursor().execute(
            "UPDATE tile_files SET filepath = ?, status = 'downloaded' WHERE filepath = ?",
            (new_filepath, old_filepath),
        )

    def get_tile_geometries(
        self,
        geometry=None,
        start_date=None,
        end_date=None,
        min_overlap=0.0,
        statuses=None,
    ):
        """
        Returns tile geometries as a GeoDataFrame with optional filters.

        Args:
            geometry (str, optional): GeoJSON string to filter by spatial overlap.
            start_date (str, optional): Start of capture_time range (inclusive).
            end_date (str, optional): End of capture_time range (inclusive).
            min_overlap (float, optional): Minimum required spatial overlap ratio.
            statuses (set[str], optional): Tile file statuses to filter by (e.g. {"downloaded"}).

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame of tile geometries.
        """
        filters = []
        params = []

        # Optional JOIN if filtering by status
        join_clause = ""
        if statuses:
            placeholders = ", ".join(["?"] * len(statuses))
            join_clause = "JOIN tile_files tf ON t.tile_id = tf.tile_id"
            filters.append(f"tf.status IN ({placeholders})")
            params.extend(statuses)

        if start_date and end_date:
            filters.append("t.capture_time BETWEEN ? AND ?")
            params.extend([start_date, end_date])

        if geometry:
            filters.append(
                "ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) >= ?"
            )
            params.extend([geometry, min_overlap])

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        sql = f"""
        SELECT t.tile_id, t.capture_time, ST_AsWKB(t.geometry) AS geom
        FROM tiles t
        {join_clause}
        {where_clause}
        """

        cursor = self.db.get_cursor()
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        print(f"rows: {len(rows)}")
        # Example ROW
        # ('20250303_190235_45_2511', # tile id
        #  datetime.datetime(2025, 3, 3, 19, 2, 35), # capture_time
        #  b'\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00...\xcaEA@') # this is the WKB representation of the geometry

        gdf = gpd.GeoDataFrame(
            [
                {
                    "tile_id": row[0],  # tile id
                    "capture_time": row[1],  # capture_time
                    "geometry": wkb.loads(row[2]) if row[2] else None,  # geometry
                }
                for row in rows
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        return gdf

    def get_tile_files_by_spatial_overlap(
        self,
        geometry,
        start_date,
        end_date,
        min_overlap=0.9,
        statuses={"downloaded"},
    ):
        placeholders = ", ".join(["?"] * len(statuses))
        sql = f"""
        SELECT t.tile_id, tf.filepath
        FROM tile_files tf
        JOIN tiles t ON tf.tile_id = t.tile_id
        WHERE t.capture_time BETWEEN ? AND ?
        AND tf.status IN ({placeholders})
        AND ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) >= ?
        """
        params = [start_date, end_date, *statuses, geometry, min_overlap]
        rows = (
            self.db.get_cursor().execute(sql, params).fetchall()
        )  # rows in format [(tile_id, filepath), ...]
        tile_dict = {}
        for tile_id, filepath in rows:
            tile_dict.setdefault(tile_id, []).append(filepath)

        return tile_dict  # in format {tile_id1: [filepath1, filepath2], tile_id2: [...], ...}

    def get_tile_ids_by_spatial_overlap(
        self,
        geometry,
        start_date,
        end_date,
        min_overlap=0.9,
        statuses={"downloaded"},
    ):
        placeholders = ", ".join(["?"] * len(statuses))
        sql = f"""
        SELECT DISTINCT t.tile_id
        FROM tile_files tf
        JOIN tiles t ON tf.tile_id = t.tile_id
        WHERE t.capture_time BETWEEN ? AND ?
        AND tf.status IN ({placeholders})
        AND ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) >= ?
        """
        params = [start_date, end_date, *statuses, geometry, min_overlap]
        tile_ids = (
            self.db.get_cursor().execute(sql, params).fetchall()
        )  # in format [('tile_id1',), ('tile_id2',), ...)]
        return list(
            map(lambda x: x[0], tile_ids)
        )  # in format ['tile_id1', 'tile_id2', ...]

    def query_ids_all_matching_tiles_and_rois(
        self, geometry, start_date, end_date, min_overlap=0.9, statuses={"downloaded"}
    ):
        placeholders = ", ".join(["?"] * len(statuses))

        sql = f"""
        WITH roi_matches AS (
            SELECT rd.tile_id
            FROM roi_files rf
            JOIN roi_data rd ON rf.roi_id = rd.roi_id AND rf.tile_id = rd.tile_id
            WHERE rd.capture_time BETWEEN ? AND ?
            AND rf.status IN ({placeholders})
            AND ST_Area(ST_Intersection(rd.intersection, ST_GeomFromGeoJSON(?))) / ST_Area(rd.intersection) >= ?
        ),
        tile_matches AS (
            SELECT t.tile_id
            FROM tile_files tf
            JOIN tiles t ON tf.tile_id = t.tile_id
            WHERE t.capture_time BETWEEN ? AND ?
            AND tf.status IN ({placeholders})
            AND ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) >= ?
        )
        SELECT DISTINCT tile_id FROM roi_matches
        UNION
        SELECT DISTINCT tile_id FROM tile_matches
        """

        params = [
            start_date,
            end_date,
            *statuses,
            geometry,
            min_overlap,  # roi_matches
            start_date,
            end_date,
            *statuses,
            geometry,
            min_overlap,  # tile_matches
        ]

        tiles_list = self.db.get_cursor().execute(sql, params).fetchall()
        return list(
            map(lambda x: x[0], tiles_list)
        )  # in format ['tile_id1', 'tile_id2', ...]

    def fetch_matching_tile_and_roi_geometries(
        self,
        geometry,
        start_date,
        end_date,
        min_overlap=0.9,
        statuses={"downloaded"},
    ):
        """
        Queries the database for geometries (tiles and ROIs) that spatially overlap with a given geometry
        within a specified date range and with a minimum overlap threshold.
        This method searches both ROI (region of interest) and tile records, filtering by capture time,
        status, and spatial overlap. It returns a GeoDataFrame containing the matching geometries.
        Args:
            geometry (str): A GeoJSON string representing the geometry to match against.
            start_date (str or datetime): The start of the capture time range (inclusive).
            end_date (str or datetime): The end of the capture time range (inclusive).
            min_overlap (float, optional): Minimum required overlap ratio (between 0 and 1) between
                the input geometry and candidate geometries. Defaults to 0.9.
            statuses (set or iterable of str, optional): Set of status values to filter the files
                (e.g., {"downloaded"}). Defaults to {"downloaded"}.
        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame with columns:
                - tile_id: ID of the tile.
                - roi_id: ID of the ROI (or None for tile-only matches).
                - capture_time: The capture time of the geometry.
                - geometry: The matched geometry as a shapely object.
        Notes:
            - The coordinate reference system (CRS) of the returned GeoDataFrame is set to EPSG:4326.
            - The function assumes the database supports spatial SQL functions such as ST_Area,
              ST_Intersection, and ST_GeomFromGeoJSON.
        """
        placeholders = ", ".join(["?"] * len(statuses))

        sql = f"""
        WITH roi_matches AS (
            SELECT rd.tile_id, rd.roi_id, rd.capture_time, rd.intersection AS geom
            FROM roi_files rf
            JOIN roi_data rd ON rf.roi_id = rd.roi_id AND rf.tile_id = rd.tile_id
            WHERE rd.capture_time BETWEEN ? AND ?
            AND rf.status IN ({placeholders})
            AND ST_Area(ST_Intersection(rd.intersection, ST_GeomFromGeoJSON(?))) / ST_Area(rd.intersection) >= ?
        ),
        tile_matches AS (
            SELECT t.tile_id, NULL AS roi_id, t.capture_time, t.geometry AS geom
            FROM tile_files tf
            JOIN tiles t ON tf.tile_id = t.tile_id
            WHERE t.capture_time BETWEEN ? AND ?
            AND tf.status IN ({placeholders})
            AND ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) >= ?
        )
        SELECT tile_id, roi_id, capture_time, ST_AsWKB(geom) AS geom FROM roi_matches
        UNION ALL
        SELECT tile_id, roi_id, capture_time, ST_AsWKB(geom) AS geom FROM tile_matches
        """

        params = [
            start_date,
            end_date,
            *statuses,
            geometry,
            min_overlap,  # roi_matches
            start_date,
            end_date,
            *statuses,
            geometry,
            min_overlap,  # tile_matches
        ]

        cursor = self.db.get_cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        # Build GeoDataFrame
        df = gpd.GeoDataFrame(
            [
                {
                    "tile_id": row[0],
                    "roi_id": row[1],
                    "capture_time": row[2],
                    "geometry": wkb.loads(row[3]) if row[3] is not None else None,
                }
                for row in rows
            ],
            geometry="geometry",
            crs="EPSG:4326",  # Adjust if your geometries use another CRS
        )

        return df

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

    def query_combined_filepaths(
        self, geometry, start_date, end_date, min_overlap=0.9, statuses={"downloaded"}
    ):
        placeholders = ", ".join(["?"] * len(statuses))

        sql = f"""
        WITH roi_matches AS (
            SELECT rf.filepath AS filepath, 'roi' AS source,
                ST_Area(ST_Intersection(rd.intersection, ST_GeomFromGeoJSON(?))) / ST_Area(rd.intersection) AS overlap
            FROM roi_files rf
            JOIN roi_data rd ON rf.roi_id = rd.roi_id AND rf.tile_id = rd.tile_id
            WHERE rd.capture_time BETWEEN ? AND ?
            AND rf.status IN ({placeholders})
        ),
        tile_matches AS (
            SELECT tf.filepath AS filepath, 'tile' AS source,
                ST_Area(ST_Intersection(t.geometry, ST_GeomFromGeoJSON(?))) / ST_Area(t.geometry) AS overlap
            FROM tile_files tf
            JOIN tiles t ON tf.tile_id = t.tile_id
            WHERE t.capture_time BETWEEN ? AND ?
            AND tf.status IN ({placeholders})
        ),
        preferred AS (
            SELECT filepath, source FROM roi_matches WHERE overlap >= ?
            UNION
            SELECT tm.filepath, tm.source FROM tile_matches tm
            WHERE overlap >= ?
            AND NOT EXISTS (
                SELECT 1 FROM roi_matches rm WHERE rm.filepath = tm.filepath AND rm.overlap >= ?
            )
        )
        SELECT DISTINCT filepath, source FROM preferred
        """

        params = [
            geometry,
            start_date,
            end_date,
            *statuses,  # roi_matches
            geometry,
            start_date,
            end_date,
            *statuses,  # tile_matches
            min_overlap,
            min_overlap,
            min_overlap,
        ]

        filepaths = self.db.get_cursor().execute(sql, params).fetchall()
        return list(
            map(lambda x: x[0], filepaths)
        )  # in format ['filepath1', 'filepath2', ...]

    def filter_out_existing_tile_ids(self, tile_ids):
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
