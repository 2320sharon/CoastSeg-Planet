import json
import os
from typing import Any, Dict, List, Optional
import asyncio
import planet
import geopandas as gpd
from planet.clients.orders import OrderStates
from coastseg_planet.orders import Order, OrderConfig

# from coastseg_planet.order_request import OrderRequest,load_request_from_config

from coastseg_planet import download
import logging
from coastseg_planet.config import TILE_STATUSES, CLOUD_COVER


# @todo make this able to add tools to the orde
def prepare_order(
    order_name: str,
    roi_path: str,
    roi_id: str,
    start_date: str,
    end_date: str,
    destination: str,
    cloud_cover: float = 0.70,
    min_area_percentage: float = 0.7,
    coregister: bool = False,
    coregister_id: str = "",
    continue_existing: bool = False,
    month_filter: List[str] = None,
) -> Order:
    """
    Creates and returns an Order object after validating all input parameters.

    Raises:
        OrderValidationError or its subclasses if validation fails.
    """
    orderconfig = OrderConfig(
        order_name=order_name,
        roi_path=roi_path,
        roi_id=roi_id,
        start_date=start_date,
        end_date=end_date,
        destination=destination,
        cloud_cover=cloud_cover,
        min_area_percentage=min_area_percentage,
        coregister=coregister,
        coregister_id=coregister_id,
        continue_existing=continue_existing,
        month_filter=month_filter or [f"{i:02}" for i in range(1, 13)],
    )

    order = Order(orderconfig)
    # use the default tools
    order.validate()

    return order


def get_tool_from_order(order: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Retrieves the first tool from the order whose key includes the specified tool_name.

    Args:
        order (Dict[str, Any]): A dictionary containing an order with a "tools" key,
            which is a list of single-key dictionaries representing tools.
        tool_name (str): The name (or substring) of the tool to search for.

    Returns:
        Dict[str, Any]: The first matching tool dictionary, or an empty dictionary if no match is found.

    Example:
        >>> order = {
        ...     "tools": [
        ...         {"clip": {"aoi": {"type": "Polygon", "coordinates": [...]}}},
        ...         {"toar": {"scale_factor": 10000}}
        ...     ]
        ... }
        >>> get_tool_from_order(order, "clip")
        {{"aoi": {"type": "Polygon", "coordinates": [...]}}}
    """
    for tool in order.get("tools", []):
        if tool_name in tool:
            return tool[tool_name]
    return {}


class OrderManager:
    def __init__(self, client):
        self.client = client
        self.orders = []

    def add_order(self, order: Order):
        self.orders.append(order)

    def add_order_from_config(self, config_path: str):
        """Load orders from a config file and add them to the manager.

        Args:
            config_path (str): Path to the config file.
                This is expected to be a JSON or YAML file containing order configurations.

        Example:
            1. JSON file:
                {
                "order_name": "sample_order",
                "roi_path": "data/sample_roi.geojson",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "destination": "output/",
                "cloud_cover": 0.6,
                "min_area_percentage": 0.8,
                "coregister": true,
                "coregister_id": "previous_order_id",
                "continue_existing": false,
                "month_filter": ["01", "02", "03", "04"],
                "tools": {
                    "clip": true,
                    "toar": true,
                    "coregister": true,
                    "coregister_id": "previous_order_id"
                }
                }

            2. YAML file:
                order_name: sample_order
                roi_path: data/sample_roi.geojson
                start_date: "2024-01-01"
                end_date: "2024-12-31"
                destination: output/
                cloud_cover: 0.6
                min_area_percentage: 0.8
                coregister: true
                coregister_id: previous_order_id
                continue_existing: false
                month_filter:
                - "01"
                - "02"
                - "03"
                - "04"
                tools:
                clip: true
                toar: true
                coregister: true
                coregister_id: previous_order_id

        """
        # creates an order object from the config file and adds it to the order manager
        order = Order.from_file(config_path)
        self.add_order(order)

    def validate_all(self):
        for order in self.orders:
            try:
                order.validate()
                print(f"✅ Order '{order.name}' is valid.")
            except Exception as e:
                print(f"❌ Order '{order.name}' failed validation: {e}")

    def get_orders(self):
        return self.orders

    def filter_by_date(self, start_date: str, end_date: str):
        """Return orders where order.start_date and order.end_date are within the given range."""
        from datetime import datetime

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        return [
            order
            for order in self.orders
            if start <= datetime.strptime(order.start_date, "%Y-%m-%d") <= end
            and start <= datetime.strptime(order.end_date, "%Y-%m-%d") <= end
        ]

    def summarize(self):
        print(f"📦 Total orders loaded: {len(self.orders)}")
        tools_summary = {"clip": 0, "toar": 0, "coregister": 0}
        for order in self.orders:
            for tool in tools_summary:
                if order.tools.get(tool):
                    tools_summary[tool] += 1
        print("🛠️  Tools usage:")
        for k, v in tools_summary.items():
            print(f"   - {k}: {v}")

    def to_dict_list(self):
        return [order.to_dict() for order in self.orders]

    def __len__(self):
        return len(self.orders)

    def clear_orders(self):
        """Remove all orders from the manager."""
        self.orders.clear()
        print("🗑️  All orders have been cleared.")

    def remove_order_by_name(self, order_name: str):
        """Remove the first order that matches the given name."""
        original_len = len(self.orders)
        self.orders = [order for order in self.orders if order.name != order_name]
        if len(self.orders) < original_len:
            print(f"🗑️  Order '{order_name}' removed.")
        else:
            print(f"⚠️  No order found with name '{order_name}'.")

    def remove_orders_by_names(self, names: list):
        """Remove all orders with names in the given list."""
        names_set = set(names)
        before_count = len(self.orders)
        self.orders = [order for order in self.orders if order.name not in names_set]
        removed_count = before_count - len(self.orders)

        if removed_count > 0:
            print(f"🗑️  Removed {removed_count} order(s): {', '.join(names_set)}")
        else:
            print("⚠️  No matching orders found to remove.")
        return removed_count

    async def await_order(
        self,
        request,
    ):
        """
        Creates an order using the provided client and request.
        The order is created and the state is updated using exponential backoff.
        (Note this does not download the order, it only creates it)

        Args:
            request (dict): The request object used to create the order.

        Returns:
            dict: The order details.
        """
        # @todo remove this and uncomment the code below
        await asyncio.sleep(1)  # Simulate some processing time
        return None

        # # First create the order and wait for it to be created
        # with planet.reporting.StateBar(state="creating") as reporter:
        #     logging.info(f"Order {order['name']} created. Waiting for completion...")
        #     order = await self.client.create_order(request)
        #     reporter.update(state="created", order_id=order["id"])
        #     await download.wait_with_exponential_backoff(client, order["id"], callback=reporter.update_state)
        # return order

    async def get_existing_order(self, order_name: str, order_states: list[str]):
        """
        Checks for an existing order by name; if none found, creates a new one.
        If an order with the same name exists, it will be returned.

        Parameters:
            order_name (str): The name of the order to check for.
            order_states (list[str]): A list of order states to filter by.
            Example: ["success", "running"]

        """
        order_ids = await download.get_order_ids_by_name(
            self.client, order_name, states=order_states
        )
        if not order_ids:
            return None
        return await self.client.get_order(order_ids[0])

    async def create_order(self, request):
        """
        Creates a new order using the provided request object.
        """
        return await self.client.create_order(request)

    def extract_geometry(self, order):
        """
        Retrieves the AOI geometry from the order if the 'clip' tool was used.
        """
        clip_tool = get_tool_from_order(order, "clip")
        if clip_tool:
            aoi = clip_tool.get("aoi")
            if aoi:
                return aoi
        # If the 'clip' tool is not found, return None
        return None

    def get_tool_from_order(
        self, order: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """
        Retrieves the first tool from the order whose key includes the specified tool_name.

        Args:
            order (Dict[str, Any]): A dictionary containing an order with a "tools" key,
                which is a list of single-key dictionaries representing tools.
            tool_name (str): The name (or substring) of the tool to search for.

        Returns:
            Dict[str, Any]: The first matching tool dictionary, or an empty dictionary if no match is found.

        Example:
            >>> order = {
            ...     "tools": [
            ...         {"clip": {"aoi": {"type": "Polygon", "coordinates": [...]}}},
            ...         {"toar": {"scale_factor": 10000}}
            ...     ]
            ... }
            >>> get_tool_from_order(order, "clip")
            {{"aoi": {"type": "Polygon", "coordinates": [...]}}}
        """
        for tool in order.get("tools", []):
            if tool_name in tool:
                return tool[tool_name]
        return {}
