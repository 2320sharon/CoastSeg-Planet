from coastseg_planet import download
from planet import Auth
import os
import asyncio
import json
import geopandas as gpd
from coastseg_planet.orders import Order

# 0. Enter the maximum cloud cover percentage (optional, default is 0.80)
CLOUD_COVER = 0.70

# 1. Select a start and end date YYYY-MM-DD for each of the orders
# 2. name the order
# Either enter the name of an existing order or create a new one
# 3. insert path to roi geojson
# roi_path = r"C:\development\coastseg-planet\CoastSeg-Planet\boardwalk\roi.geojson"



start_date = "2024-07-15"
end_date = "2024-07-31"
roi_path = r"C:\development\coastseg-planet\CoastSeg-Planet\5_geojson\3_DUCK\roi.geojson"
order_name= f"DUCK_cloud_{CLOUD_COVER}_TOAR_enabled_{start_date}_to_{end_date}_coregistered_with_20240728_150711_16_24bf"
# Create multiple orders
order = Order(
    order_name= order_name,
    roi_path=roi_path,
    start_date=start_date,
    end_date=end_date,
    cloud_cover=CLOUD_COVER,
    destination=os.path.join(os.getcwd(), "downloads", order_name),
    continue_existing=False,
    coregister_id="20240728_150711_16_24bf",
    coregister=True,
    min_area_percentage=0.7,

).get_order()

# Change the order name and roi path for each order
start_date = "2024-07-14"
end_date = "2024-07-31"
roi_path = r"C:\development\coastseg-planet\CoastSeg-Planet\5_geojson\1_SANTA_CRUZ\roi.geojson"
order_name= f"santa_cruz_cloud_{CLOUD_COVER}_TOAR_enabled_{start_date}_to_{end_date}_coregistered_with_20240723_181417_27_24c8"
order2 = Order(
    order_name= order_name,
    roi_path=roi_path,
    start_date=start_date,
    end_date=end_date,
    cloud_cover=CLOUD_COVER,
    destination=os.path.join(os.getcwd(), "downloads", order_name),
    continue_existing=False,
    coregister_id="20240723_181417_27_24c8",
    coregister=True,
    min_area_percentage=0.7,

).get_order()

# make the list of orders
order_list = [order, order2]



# 4. read the api key from the config file and set it in the environment
# Enter the API key into config.ini with the following format
# [DEFAULT]
# API_KEY = <PLANET API KEY>

config_filepath = os.path.join(os.getcwd(),"config.ini")
if os.path.exists(config_filepath) is False:
    raise FileNotFoundError(f"Config file not found at {config_filepath}")
config = download.read_config(config_filepath)
# set the API key in the environment and store it
if config.get("DEFAULT","API_KEY") == "":
    raise ValueError("API_KEY not found in config file. Please enter your API key in the config file and try again")
os.environ["API_KEY"] = config["DEFAULT"]["API_KEY"]
auth = Auth.from_env("API_KEY")
auth.store()


asyncio.run(download.download_multiple_orders_in_parallel(order_list))