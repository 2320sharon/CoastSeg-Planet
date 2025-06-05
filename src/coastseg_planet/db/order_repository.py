# db/order_repository.py

import json
from coastseg_planet.db.base import BaseDuckDB


class OrderRepository:
    """
    Handles all logic for inserting or updating entries in the 'orders' table.

    Attributes:
        db (BaseDuckDB): Shared database connection.
    """

    def __init__(self, base: BaseDuckDB):
        self.db = base

    def insert_order(
        self, order_id, filename, status=None, filepath=None, geometry=None
    ):
        cursor = self.db.get_cursor()
        geom_param = json.dumps(geometry) if geometry else None

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
        self.db.commit()
