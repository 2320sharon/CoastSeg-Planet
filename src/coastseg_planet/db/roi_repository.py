# db/roi_repository.py

import json
from coastseg_planet.db.base import parse_capture_time, BaseDuckDB


class ROIRepository:
    """
    Manages all database operations related to the 'roi_data' and 'roi_files' tables.

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
    ):
        """
        Inserts or updates a ROI record in 'roi_data'.

        Args:
            roi_id (str): ROI identifier.
            tile_id (str): Tile identifier.
            capture_time (str): Capture timestamp in 'YYYYMMDD_HHMMSS_XX' format.
            geom (dict, optional): Full ROI geometry (GeoJSON).
            intersection (dict, optional): Tile-clipped geometry.
        """
        cursor = self.db.get_cursor()
        parsed_time = parse_capture_time(capture_time)

        # Determine geometry to insert
        intersection_geojson = (
            json.dumps(intersection)
            if intersection
            else (json.dumps(geom) if geom else None)
        )
        geometry_geojson = json.dumps(geom) if geom else None

        cursor.execute(
            """
            INSERT INTO roi_data (roi_id, tile_id, capture_time, geometry, intersection)
            VALUES (?, ?, ?, ST_GeomFromGeoJSON(?), ST_GeomFromGeoJSON(?))
            ON CONFLICT(roi_id, tile_id) DO UPDATE SET
                capture_time = EXCLUDED.capture_time,
                geometry = EXCLUDED.geometry,
                intersection = EXCLUDED.intersection;
            """,
            (roi_id, tile_id, parsed_time, geometry_geojson, intersection_geojson),
        )

        self.db.commit()
        print(f"[SUCCESS] ROI {roi_id} inserted/updated for tile {tile_id}.")

    def query_roi_tiles_by_geometry(
        self,
        geometry,
        start_date,
        end_date,
        min_overlap=0.9,
        statuses={"downloaded"},
    ):
        cursor = self.db.get_cursor()
        placeholders = ", ".join(["?"] * len(statuses))
        sql = f"""
        SELECT DISTINCT rd.tile_id
        FROM roi_files rf
        JOIN roi_data rd ON rf.roi_id = rd.roi_id AND rf.tile_id = rd.tile_id
        WHERE rd.capture_time BETWEEN ? AND ?
        AND rf.status IN ({placeholders})
        AND ST_Area(ST_Intersection(rd.intersection, ST_GeomFromGeoJSON(?))) / ST_Area(rd.intersection) >= ?
         """
        params = [start_date, end_date, *statuses, geometry, min_overlap]
        tile_ids = cursor.execute(
            sql, params
        ).fetchall()  # in format [('tile_id1',), ('tile_id2',), ...)]
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

    def update_roi(self, roi_id, tile_id, geom=None, intersection=None):
        """
        Updates an existing ROI entry in 'roi_data'. Only non-None fields will be updated.

        Args:
            roi_id (str): The ROI identifier to update.
            tile_id (str): Tile identifier.
            geom (dict, optional): New GeoJSON geometry. If None, geometry is not modified.
            intersection (dict, optional): New intersection geometry.
        """
        cursor = self.db.get_cursor()

        # Check if ROI exists before updating
        cursor.execute(
            "SELECT 1 FROM roi_data WHERE roi_id = ? AND tile_id = ?", (roi_id, tile_id)
        )
        if not cursor.fetchone():
            print(
                f"[ERROR] ROI entry (roi_id={roi_id}, tile_id={tile_id}) does not exist."
            )
            return

        updates = []
        params = []

        if geom is not None:
            updates.append("geometry = ST_GeomFromGeoJSON(?)")
            params.append(json.dumps(geom))
        if intersection is not None:
            updates.append("intersection = ST_GeomFromGeoJSON(?)")
            params.append(json.dumps(intersection))

        if not updates:
            print(f"[INFO] No updates provided for ROI {roi_id}.")
            return

        sql = (
            f"UPDATE roi_data SET {', '.join(updates)} WHERE roi_id = ? AND tile_id = ?"
        )
        params.extend([roi_id, tile_id])
        cursor.execute(sql, params)
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
        """
        Retrieve ROI file metadata by status and optionally by order.

        Args:
            status (str or list): Status or list of statuses to filter by.
            order_id (str, optional): Order ID to filter by.

        Returns:
            List[dict]: Resulting rows.
        """
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

        query = f"""
        SELECT {query_columns}
        FROM roi_files
        WHERE {' AND '.join(where_clauses)}
        """
        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
