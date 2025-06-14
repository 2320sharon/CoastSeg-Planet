from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os
from typing import List
import json
import yaml
from shapely.geometry import shape, Polygon
from typing import Union, Set, Dict


# ---------------------
# Enum for Order Status
# ---------------------
class OrderStatus(Enum):
    NOT_FOUND = "NotFound"  # Order not yet created
    RUNNING = "Running"  # Being created, not ready
    SUCCESSFUL = "Successful"  # Ready for download
    UNAVAILABLE = "Unavailable"  # Cannot be found or accessed


# ---------------------
# Custom Exceptions
# ---------------------
class OrderValidationError(Exception):
    pass


class ROIFileNotFound(OrderValidationError):
    pass


class InvalidDateFormat(OrderValidationError):
    pass


class DateRangeError(OrderValidationError):
    pass


class CloudCoverError(OrderValidationError):
    pass


class AreaPercentageError(OrderValidationError):
    pass


class DestinationError(OrderValidationError):
    pass


class GeoJSONValidationError(OrderValidationError):
    pass


# ---------------------
# Order Configuration Dataclass
# This class is used to accept all the parameters necessary to create an order.
# It is used as the input to the Order class.
# ---------------------
@dataclass
class OrderConfig:
    """Configuration for the order.
    order_name: str - Name of the order.
    roi_path: str - Path to the ROI file (GeoJSON).
    start_date: str - Start date for the order (YYYY-MM-DD).
    end_date: str - End date for the order (YYYY-MM-DD).
    destination: str - Destination folder for the order.
    cloud_cover: float - Maximum cloud cover percentage (0.0 to 1.0).
    min_area_percentage: float - Minimum area percentage for the order (0.0 to 1.0).
    coregister: bool - Whether to coregister the order.
    coregister_id: str - ID of the coregistered order.
    continue_existing: bool - Whether to continue an existing order.
    month_filter: List[str] - List of months to filter the order (01 to 12).
    product_bundle: str - Product bundle to use (default is "analytic_udm2").
    item_type: str - Item type to use (default is "PSScene").
    tools: Union[Set[str], Dict[str, bool]] - Tools to apply to the order.
    - If a set, it contains tool names like "clip", "toar".
    - If a dict, it contains tool names as keys and boolean values indicating whether to apply the tool.

    """

    order_name: str
    roi_path: str
    roi_id: str
    start_date: str
    end_date: str
    destination: str
    cloud_cover: float = 0.70
    min_area_percentage: float = 0.7
    coregister: bool = False
    coregister_id: str = ""
    continue_existing: bool = False
    month_filter: List[str] = field(
        default_factory=lambda: [f"{i:02}" for i in range(1, 13)]
    )
    product_bundle: str = "analytic_udm2"
    item_type: str = "PSScene"
    tools: Union[Set[str], Dict[str, bool]] = field(default_factory=set)


# ---------------------
# Main Order Class
# ---------------------
class Order:
    """
    Represent an order that can be placed with the Planet Labs API.

    This class creates and validates a configuration dictionary containing all information
    necessary to place a new order with Planet Labs or retrieve details about an existing one.

    Responsibilities:
    - Validate input fields such as ROI path, date range, cloud cover, and destination
    - Ensure the ROI GeoJSON is valid and contains Polygon geometries
    - Track the lifecycle status of an order (e.g., not found, running, successful)

    Attributes:
        name (str): Name of the order.
        config (OrderConfig): All order parameters needed for the API in a single dataclass.

        status (OrderStatus): Current state of the order lifecycle.

            "Available" indicates that the order is ready for download.
            "Running" indicates that the order is being created and is not yet ready.
            "NotFound" indicates that the order has not been created yet.
            "Unavailable" indicates that the order cannot be found or accessed.

        available (bool): Indicates if the order is available to download or not.
        tools (dict): Dictionary of tools and their settings for the order.
            - clip (bool): Whether to clip the images to the ROI.
            - toar (bool): Whether to apply TOAR (Top of Atmosphere Reflectance).
            - coregister (bool): Whether to coregister the images.
            - coregister_id (str): ID of the coregistered order.


    Methods:
        validate(): Runs all validation checks.
        to_dict(): Returns the order configuration as a dictionary for use with the API.
        from_dict(): Creates an Order object from a dictionary.
        from_file(): Loads order config from a YAML or JSON file.
        __repr__(): Displays a human-readable summary of the order.

    """

    # default_tools = {
    #     "clip": True,
    #     "toar": True,
    #     "coregister": False,
    #     "coregister_id": "",
    # }

    def __init__(self, config: OrderConfig):
        self.config = config
        self._status = OrderStatus.NOT_FOUND
        self._available = False
        self._tools = self.set_tools(config.tools)

        self.validate()

        self._status = OrderStatus.SUCCESSFUL
        self._available = True

    def set_tools(self, tools: Union[Set[str], Dict[str, bool]]):
        """
        Set the tools for the order.
        Args:
            tools (Union[Set[str], Dict[str, bool]]): Tools to apply to the order.
                - If a set, it contains tool names like "clip", "toar".
                - If a dict, it contains tool names as keys and boolean values indicating whether to apply the tool.
        Returns:
            dict: A dictionary of tools with their settings.
        Raises:
            TypeError: If tools is not a dict, set, or None.
        """
        if isinstance(tools, set):
            return {**{tool: True for tool in tools}}
        elif isinstance(tools, dict):
            return {**tools}
        elif tools is None:
            return {}
        elif isinstance(tools, list):
            return {**{tool: True for tool in tools}}
        else:
            raise TypeError("tools must be a dict, set, or None")

    def __repr__(self):
        return (
            f"<Order(name={self.name}, status={self.status.name}, "
            f"available={self.available})> dates: {self.dates}, "
        )

    def to_dict(self) -> dict:
        """
        Returns the order configuration as a dictionary.

        Example output:
        {
            "order_name": "example_order",
            "roi_path": "path/to/roi.geojson",
            "roi_id": "roi_777",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "cloud_cover": 0.7,
            "min_area_percentage": 0.7,
            "coregister": False,
            "coregister_id": "",
            "destination": "./output",
            "continue_existing": False,
            "month_filter": ["01", "02", "03", ..., "12"],
            "product_bundle": "analytic_udm2",
            "item_type": "PSScene",
            "tools": {"clip": True, "toar": True, "coregister": False, "coregister_id": ""}
        }
        """
        result = self.config.__dict__.copy()
        result["tools"] = self.tools
        return self.config.__dict__.copy()

    @classmethod
    def from_dict(cls, config: dict):
        """
        Create an Order instance from a dictionary containing configuration data.

        Args:
            config (dict): A dictionary with keys matching the fields of OrderConfig.

        Returns:
            Order: A new validated Order instance based on the provided configuration.

        Raises:
            KeyError, OrderValidationError: If required keys are missing or validation fails.
        """
        order = cls(
            OrderConfig(
                order_name=config["order_name"],
                roi_path=config["roi_path"],
                roi_id=config.get("roi_id", ""),
                start_date=config["start_date"],
                end_date=config["end_date"],
                destination=config["destination"],
                cloud_cover=config.get("cloud_cover", 0.7),
                min_area_percentage=config.get("min_area_percentage", 0.7),
                coregister=config.get("coregister", False),
                coregister_id=config.get("coregister_id", ""),
                continue_existing=config.get("continue_existing", False),
                month_filter=config.get(
                    "month_filter",
                    [f"{i:02}" for i in range(1, 13)],
                ),
                tools=config.get("tools", {}),
            ),
        )
        order.validate()
        return order

    def update_config(self, updates: dict):
        """
        Update the configuration of the order.

        Args:
            updates (dict): Dictionary of config fields to update.

        Raises:
            AttributeError: If an invalid config field is passed.
            OrderValidationError: If validation fails after update.
        """
        for key, value in updates.items():
            if not hasattr(self.config, key):
                raise AttributeError(f"'OrderConfig' has no attribute '{key}'")
            setattr(self.config, key, value)

        # Re-validate after updating
        self.validate()

    @classmethod
    def from_file(cls, config_path: str):
        """
        Load an Order instance from a YAML or JSON configuration file.

        This method parses the file, builds a configuration dictionary,
        and initializes an Order instance using from_dict.

        Args:
            config_path (str): Path to a .json, .yaml, or .yml config file.

        Returns:
            Order: A validated Order instance.

        Raises:
            ValueError: If the file extension is unsupported.
            FileNotFoundError, json.JSONDecodeError, yaml.YAMLError: On file issues.

        Example:
            >>> order = Order.from_file("config/order_request.yaml")
            >>> print(order.name)
            >>> print(order.tools)
        """

        if config_path.endswith(".json"):
            with open(config_path) as f:
                config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(
                "Unsupported config file type. Must be .json, .yaml, or .yml"
            )

        order = cls.from_dict(config)
        order.validate()
        return order

    # Have get order do the same thing as to_dict
    def get_order(self) -> dict:
        """
        Returns the order configuration as a dictionary.
        """
        return self.to_dict()

    # for prep_order make it do the validation and return the order
    def prep_order(self) -> dict:
        """
        Prepares the order by validating and returning the order configuration.
        """
        self.validate()
        return self.to_dict()

    @property
    def tools(self):
        """
        Returns the current tools configuration.
        """
        return self._tools

    def get_roi_geodataframe(self):
        """
        Load the ROI from the configured GeoJSON path and return it as a GeoDataFrame
        in EPSG:4326 coordinate reference system.

        Returns:
            geopandas.GeoDataFrame: ROI geometry as GeoDataFrame in EPSG:4326.

        Raises:
            ROIFileNotFound: If the ROI path does not exist.
            GeoJSONValidationError: If the file cannot be read or is invalid.
        """
        import geopandas as gpd

        path = self.config.roi_path
        if not os.path.exists(path):
            raise ROIFileNotFound(f"ROI path '{path}' does not exist.")

        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            raise GeoJSONValidationError(f"Failed to load GeoJSON as GeoDataFrame: {e}")

        return gdf.to_crs(epsg=4326)

    def update_tool(self, name: str, value):
        """
        Update an individual tool setting.

        Args:
            name (str): Tool name to update.
            value (any): New value for the tool.

        Raises:
            KeyError: If the tool name is not recognized.
        """
        if name not in self._tools:
            raise KeyError(f"'{name}' is not a recognized tool.")
        self._tools[name] = value

    @property
    def name(self):
        return self.config.order_name

    @property
    def destination(self):
        """
        Returns the destination folder for the order.
        """
        return self.config.destination

    @property
    def dates(self):
        return [self.config.start_date, self.config.end_date]

    @property
    def month_filter(self):
        return self.config.month_filter

    @property
    def product_bundle(self):
        return self.config.product_bundle

    @property
    def item_type(self):
        return self.config.item_type

    @property
    def available(self):
        return self._available

    @property
    def status(self):
        return self._status

    def validate(self):
        self._validate_roi()
        self._validate_geojson()
        self._validate_destination()
        self._validate_dates()
        self._validate_cloud_cover()
        self._validate_min_area()

    def _validate_roi(self):
        print(f"Validating ROI path: {self.config.roi_path}")
        if not os.path.exists(self.config.roi_path):
            raise ROIFileNotFound(f"ROI path '{self.config.roi_path}' does not exist.")

    def _validate_geojson(self):
        path = self.config.roi_path
        if not path.lower().endswith(".geojson"):
            raise GeoJSONValidationError(
                f"ROI path '{path}' is not a valid GeoJSON file."
            )

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise GeoJSONValidationError(f"Error parsing GeoJSON file: {e}")

        if "features" not in data:
            raise GeoJSONValidationError(
                f"GeoJSON file '{path}' does not contain 'features'."
            )

        for feature in data["features"]:
            geom = shape(feature["geometry"])
            if not isinstance(geom, Polygon):
                raise GeoJSONValidationError(
                    f"Expected Polygon, but found {geom.geom_type}"
                )

    def _validate_destination(self):
        if not os.path.exists(self.config.destination):
            os.makedirs(self.config.destination, exist_ok=True)
        elif not os.path.isdir(self.config.destination):
            raise DestinationError(
                f"Destination '{self.config.destination}' is not a directory."
            )

    def _validate_dates(self):
        try:
            start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        except ValueError:
            raise InvalidDateFormat("Dates must be in YYYY-MM-DD format.")

        if start >= end:
            raise DateRangeError("Start date must be before end date.")

    def _validate_cloud_cover(self):
        val = self.config.cloud_cover
        if not isinstance(val, float) or not (0 <= val <= 1):
            raise CloudCoverError("Cloud cover must be a float between 0 and 1.")

    def _validate_min_area(self):
        val = self.config.min_area_percentage
        if not isinstance(val, float) or not (0 <= val <= 1):
            raise AreaPercentageError(
                "Minimum area percentage must be a float between 0 and 1."
            )
