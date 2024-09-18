
import os
from datetime import datetime
from shapely.geometry import shape, Polygon
import json

class Order:
    
    def __init__(self, order_name, roi_path, start_date, end_date, destination, cloud_cover=0.70, min_area_percentage=0.7, coregister=False, coregister_id="", continue_existing=False,months_filter:list=None):
        if months_filter is None:
            months_filter = ['01','02','03','04','05','06','07','08','09','10','11','12']
        
        self.order = self.make_order_dict(
            order_name, roi_path, start_date, end_date, destination, cloud_cover, min_area_percentage, coregister, coregister_id,continue_existing,months_filter
        )
        self.prep_order()

    def get_order(self):
        return self.order
    
    def make_order_dict(self, order_name, roi_path, start_date, end_date, destination, cloud_cover=0.70, min_area_percentage=0.7, coregister=False, coregister_id="", continue_existing=False,months_filter=None):
        if months_filter is None:
            months_filter = ['01','02','03','04','05','06','07','08','09','10','11','12']
        return {
            "order_name": order_name,
            "roi_path": roi_path,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover": cloud_cover,
            "min_area_percentage": min_area_percentage,
            "coregister": coregister,
            "coregister_id": coregister_id,
            "destination": destination,
            "continue_existing": continue_existing,  
            'months_filter': months_filter           
        }
    

    def check_roi_exists(self):
        """Check if the ROI path exists."""
        if not os.path.exists(self.order['roi_path']):
            raise FileNotFoundError(f"ROI path '{self.order['roi_path']}' does not exist.")
        print(f"ROI path '{self.order['roi_path']}' exists.")

 
    def check_geojson_valid_polygon(self):
        """Check if the ROI path contains a valid GeoJSON file with a single Polygon geometry."""
        roi_path = self.order['roi_path']
        
        # Ensure the file has a .geojson extension
        if not roi_path.lower().endswith(".geojson"):
            raise ValueError(f"ROI path '{roi_path}' is not a valid GeoJSON file.")

        # Load and validate the GeoJSON
        try:
            with open(roi_path, 'r') as f:
                geojson_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing GeoJSON file: {e}")
        
        # Ensure the GeoJSON contains a valid 'Polygon' geometry
        if 'features' not in geojson_data:
            raise ValueError(f"GeoJSON file '{roi_path}' does not contain 'features'.")
        
        # Checking each feature for valid Polygon geometry
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            if isinstance(geom, Polygon):
                print(f"Valid Polygon found in GeoJSON file '{roi_path}'.")
            else:
                raise ValueError(f"Invalid geometry type in GeoJSON file '{roi_path}'. Expected Polygon, but found {geom.geom_type}.")


    def check_destination(self):
        """Make sure the destination exists or create it if it does not exist."""
        if not os.path.exists(self.order['destination']):
            os.makedirs(self.order['destination'], exist_ok=True)
            print(f"Destination '{self.order['destination']}' did not exist. It has been created.")
        else:
            if not os.path.isdir(self.order['destination']):
                raise NotADirectoryError(f"Destination '{self.order['destination']}' is not a directory.")
            print(f"Destination '{self.order['destination']}' exists.")

    def validate_date_format(self, date_text):
        """Make sure the date is in the correct format."""
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date '{date_text}' is not in the correct format. Use YYYY-MM-DD.")
        print(f"Date '{date_text}' is valid.")
    
    def check_dates(self):
        """Validate the start and end dates and ensure the start date is before the end date."""
        self.validate_date_format(self.order['start_date'])
        self.validate_date_format(self.order['end_date'])
        start_date = datetime.strptime(self.order['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.order['end_date'], '%Y-%m-%d')
        if start_date >= end_date:
            raise ValueError(f"Start date '{self.order['start_date']}' must be before the end date '{self.order['end_date']}'.")
        print(f"Start and end dates are valid. Start: {self.order['start_date']}, End: {self.order['end_date']}")

    def check_cloud_cover(self):
        """Make sure the cloud cover is a float between 0 and 1."""
        cloud_cover = self.order['cloud_cover']
        if not isinstance(cloud_cover, float) or not (0 <= cloud_cover <= 1):
            raise ValueError("Cloud cover must be a float between 0 and 1.")
        print(f"Cloud cover {cloud_cover} is valid.")

    def check_min_area_percentage(self):
        """Make sure the min_area_percentage is a float between 0 and 1."""
        min_area_percentage = self.order['min_area_percentage']
        if not isinstance(min_area_percentage, float) or not (0 <= min_area_percentage <= 1):
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


    