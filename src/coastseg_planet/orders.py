import os
from datetime import datetime
from shapely.geometry import shape, Polygon
import json


class Order:

    def __init__(
        self,
        order_name,
        roi_path,
        start_date,
        end_date,
        destination,
        cloud_cover=0.70,
        min_area_percentage=0.7,
        coregister_id="",
        continue_existing=False,
        month_filter: list = None,
        tools: set = None,
    ):
        """
        Initialize an Order object with all necessary parameters and run basic validations.

        Args:
            order_name (str): Name of the order.
            roi_path (str): Path to the ROI GeoJSON file.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            destination (str): Directory to save the processed results.
            cloud_cover (float, optional): Maximum allowed cloud cover (0.0 to 1.0). Defaults to 0.70.
            min_area_percentage (float, optional): Minimum area percentage required (0.0 to 1.0). Defaults to 0.7.
            coregister_id (str, optional): ID to use for coregistration. Defaults to an empty string.
            continue_existing (bool, optional): Whether to continue an existing order. Defaults to False.
            month_filter (list of str, optional): List of two-digit month strings to filter by (e.g. ["01", "02"]). Defaults to all months.
            tools (set or list, optional): Set or list of preprocessing tools to apply (e.g. {"clip", "toar"}). Converted to a set internally.

        Raises:
            FileNotFoundError: If the ROI path does not exist.
            ValueError: For invalid date formats or parameter values.
            NotADirectoryError: If the destination path exists but is not a directory.
        """
        if month_filter is None:
            month_filter = [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ]
        # Convert tools to set if provided as list
        if tools is not None and not isinstance(tools, set):
            tools = set(tools)
        self.order = self.make_order_dict(
            order_name,
            roi_path,
            start_date,
            end_date,
            destination,
            cloud_cover,
            min_area_percentage,
            coregister_id,
            continue_existing,
            month_filter,
            tools,
        )
        self.prep_order()

    def get_order(self):
        """
        Return a copy of the order dictionary with tools serialized as a list.

        This method ensures that the 'tools' field, which may be stored as a set
        internally, is converted to a list for compatibility with serialization formats
        such as JSON.

        Returns:
            dict: A copy of the order dictionary with all parameters and tools as a list.
        """
        # Return tools as a list for serialization compatibility
        order_copy = self.order.copy()
        if "tools" in order_copy and isinstance(order_copy["tools"], set):
            order_copy["tools"] = list(order_copy["tools"])
        return order_copy

    def make_order_dict(
        self,
        order_name,
        roi_path,
        start_date,
        end_date,
        destination,
        cloud_cover=0.70,
        min_area_percentage=0.7,
        coregister_id="",
        continue_existing=False,
        month_filter=None,
        tools=None,
    ):
        """
        Create and return a dictionary representing the order configuration.

        This method gathers all input parameters into a single dictionary structure,
        including optional filters and preprocessing tools.

        Args:
            order_name (str): Name of the order.
            roi_path (str): Path to the ROI GeoJSON file.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            destination (str): Directory to save the processed results.
            cloud_cover (float, optional): Maximum cloud cover allowed (0.0 to 1.0). Defaults to 0.70.
            min_area_percentage (float, optional): Minimum coverage area required (0.0 to 1.0). Defaults to 0.7.
            coregister_id (str, optional): ID used for coregistration reference. Defaults to an empty string.
            continue_existing (bool, optional): Flag to continue a previously started order. Defaults to False.
            month_filter (list of str, optional): List of months to include (e.g., ["01", "02"]). Defaults to all months.
            tools (set or list, optional): Set or list of tools to apply. If provided, stored as a set.

        Returns:
            dict: A dictionary containing all order parameters.
        """
        if month_filter is None:
            month_filter = [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ]
        order_dict = {
            "order_name": order_name,
            "roi_path": roi_path,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover": cloud_cover,
            "min_area_percentage": min_area_percentage,
            "coregister_id": coregister_id,
            "destination": destination,
            "continue_existing": continue_existing,
            "month_filter": month_filter,
        }
        if tools is not None:
            order_dict["tools"] = set(tools)
        return order_dict

    def check_roi_exists(self):
        """Check if the ROI path exists."""
        if not os.path.exists(self.order["roi_path"]):
            raise FileNotFoundError(
                f"ROI path '{self.order['roi_path']}' does not exist."
            )
        print(f"ROI path '{self.order['roi_path']}' exists.")

    def check_geojson_valid_polygon(self):
        """Check if the ROI path contains a valid GeoJSON file with a single Polygon geometry."""
        roi_path = self.order["roi_path"]

        # Ensure the file has a .geojson extension
        if not roi_path.lower().endswith(".geojson"):
            raise ValueError(f"ROI path '{roi_path}' is not a valid GeoJSON file.")

        # Load and validate the GeoJSON
        try:
            with open(roi_path, "r") as f:
                geojson_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing GeoJSON file: {e}")

        # Ensure the GeoJSON contains a valid 'Polygon' geometry
        if "features" not in geojson_data:
            raise ValueError(f"GeoJSON file '{roi_path}' does not contain 'features'.")

        # Checking each feature for valid Polygon geometry
        for feature in geojson_data["features"]:
            geom = shape(feature["geometry"])
            if isinstance(geom, Polygon):
                print(f"Valid Polygon found in GeoJSON file '{roi_path}'.")
            else:
                raise ValueError(
                    f"Invalid geometry type in GeoJSON file '{roi_path}'. Expected Polygon, but found {geom.geom_type}."
                )

    def check_destination(self):
        """Make sure the destination exists or create it if it does not exist."""
        if not os.path.exists(self.order["destination"]):
            os.makedirs(self.order["destination"], exist_ok=True)
            print(
                f"Destination '{self.order['destination']}' did not exist. It has been created."
            )
        else:
            if not os.path.isdir(self.order["destination"]):
                raise NotADirectoryError(
                    f"Destination '{self.order['destination']}' is not a directory."
                )
            print(f"Destination '{self.order['destination']}' exists.")

    def validate_date_format(self, date_text):
        """Make sure the date is in the correct format."""
        try:
            datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date '{date_text}' is not in the correct format. Use YYYY-MM-DD."
            )
        print(f"Date '{date_text}' is valid.")

    def check_dates(self):
        """Validate the start and end dates and ensure the start date is before the end date."""
        self.validate_date_format(self.order["start_date"])
        self.validate_date_format(self.order["end_date"])
        start_date = datetime.strptime(self.order["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(self.order["end_date"], "%Y-%m-%d")
        if start_date >= end_date:
            raise ValueError(
                f"Start date '{self.order['start_date']}' must be before the end date '{self.order['end_date']}'."
            )
        print(
            f"Start and end dates are valid. Start: {self.order['start_date']}, End: {self.order['end_date']}"
        )

    def check_cloud_cover(self):
        """Make sure the cloud cover is a float between 0 and 1."""
        cloud_cover = self.order["cloud_cover"]
        if not isinstance(cloud_cover, float) or not (0 <= cloud_cover <= 1):
            raise ValueError("Cloud cover must be a float between 0 and 1.")
        print(f"Cloud cover {cloud_cover} is valid.")

    def check_min_area_percentage(self):
        """Make sure the min_area_percentage is a float between 0 and 1."""
        min_area_percentage = self.order["min_area_percentage"]
        if not isinstance(min_area_percentage, float) or not (
            0 <= min_area_percentage <= 1
        ):
            raise ValueError("Minimum area percentage must be a float between 0 and 1.")
        print(f"Minimum area percentage {min_area_percentage} is valid.")

    def prep_order(self):
        """Run all the validation checks."""
        self.check_roi_exists()
        self.check_destination()  # This will create the destination folder if it doesn't exist
        self.check_dates()  # Ensure start date is before end date
        self.check_cloud_cover()
        self.check_min_area_percentage()
        print("Order is ready.")
