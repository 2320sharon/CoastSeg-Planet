import os
import asyncio
from coastseg_planet.orders import Order, OrderConfig

# ----------------------------
# USER CONFIGURATION SECTION
# ----------------------------

# Cloud cover threshold (maximum cloud cover percentage) (0.0 to 1.0)
CLOUD_COVER = 0.70

# Months of imagery to retain in the download
# Use two-digit month numbers (e.g., "01" for January, "02" for February, etc.)
MONTH_FILTER = ["05", "06", "07", "08", "09", "10"]

# ----------------------------
# 1. Define Orders (You can add more by copying the block)
# ----------------------------

# 1. Select a start and end date YYYY-MM-DD for each of the orders
# 2. name the order
# Either enter the name of an existing order or create a new one
# 3. insert path to roi geojson
# roi_path = r"C:\development\coastseg-planet\CoastSeg-Planet\boardwalk\roi.geojson"

# First order: DUCK
start_date = "2024-07-15" # YYYY-MM-DD
end_date = "2024-07-31"
roi_path = (
    r"C:\development\coastseg-planet\CoastSeg-Planet\5_geojson\3_DUCK\roi.geojson"
)
order_name = f"DUCK_cloud_{CLOUD_COVER}_TOAR_enabled_{start_date}_to_{end_date}"
# Create multiple orders
order_config = OrderConfig(
    order_name=order_name,
    roi_path=roi_path,
    roi_id="duck_ROI",
    start_date=start_date,
    end_date=end_date,
    cloud_cover=CLOUD_COVER,
    destination=os.path.join(os.getcwd(), "downloads", order_name),
    continue_existing=False,
    min_area_percentage=0.25,
    month_filter=MONTH_FILTER,
    tools={"clip", "toar"},  # Use clip and toar tools by default)
order1 = Order(order_config)

# Second order: SANTA_CRUZ
# Note: Make sure to change the order name and roi path for each order
start_date = "2024-07-14"
end_date = "2024-07-31"
roi_path = (
    r"C:\development\coastseg-planet\CoastSeg-Planet\5_geojson\1_SANTA_CRUZ\roi.geojson"
)
order_name = f"santa_cruz_cloud_{CLOUD_COVER}_TOAR_enabled_{start_date}_to_{end_date}"
order_config = OrderConfig(
    order_name=order_name,
    roi_path=roi_path,
    roi_id="santa_cruz_ROI",
    start_date=start_date,
    end_date=end_date,
    cloud_cover=CLOUD_COVER,
    destination=os.path.join(os.getcwd(), "downloads", order_name),
    continue_existing=False,
    min_area_percentage=0.25,
    month_filter=MONTH_FILTER,
    tools={"clip", "toar"},  # Use clip and toar tools by default
).get_order()

# I would not recommend more than 5 orders at a time. Planet API has a limit of 5 simultaneous downloads

# make the list of orders
order_list = [order1, order2]
# ----------------------------
# 2. Authenticate with Planet API
# ----------------------------

# Read the api key from the config file and set it in the environment
# Enter the API key into config.ini with the following format
# [DEFAULT]
# API_KEY = <PLANET API KEY>

config_filepath = os.path.join(os.getcwd(), "config.ini")
config_filepath = os.path.join(os.getcwd(), "config.ini")
if os.path.exists(config_filepath) is False:
    raise FileNotFoundError(f"Config file not found at {config_filepath}")
config = download.read_config(config_filepath)
# set the API key in the environment and store it
if config.get("DEFAULT",  "API_KEY") == "":
    raise ValueError(
        
        "API_KEY not found in config file. Please enter your API key in the config file and try again"
    
    )
os.environ["API_KEY"] = config["DEFAULT"]["API_KEY"]
auth = Auth.from_env("API_KEY")
auth.store()

# download the orders in parallel
asyncio.run(download.download_multiple_orders_in_parallel(order_list))
