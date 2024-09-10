from coastseg_planet import download
from planet import Auth
import os
import asyncio
import geopandas as gpd
# 0. Enter the maximum cloud cover percentage (optional, default is 0.80)
CLOUD_COVER = 0.70

# 1. Select a start and end date YYYY-MM-DD
start_date = "2020-06-01"
end_date = "2023-08-01"

# 1a. List the month numbers you want to keep. Always use two digits for the month number
month_filter=['05','06','07','08','09','10']

#1b. Choose the minimum area percentage of the ROI's area that must be covered by the images to be downloaded
# the default is 0.6 or 60% of the image must be covered by the ROI to be downloaded
min_area_percentage = 0.6

#1c. Choose whether to continue and existing order or create a new one
# Only set this to be True if you want to continue downloading an existing order with the same name that was unfinished
continue_existing = False



# 2. name the order
# Either enter the name of an existing order or create a new one
order_name = f"DUCK_pier_cloud_{CLOUD_COVER}_TOAR_enabled_{start_date}_to_{end_date}"

# 3. insert path to roi geojson
roi_path = os.path.join(os.getcwd(),"sample_data", "rois.geojson")
roi = gpd.read_file(roi_path)

# 4. read the api key from the config file and set it in the environment
# Enter the API key into config.ini with the following format
# [DEFAULT]
# API_KEY = <PLANET API KEY>

# config_filepath = os.path.join(os.getcwd(),"config.ini")
config_filepath = r"C:\development\coastseg-planet\CoastSeg-Planet\config.ini"

if os.path.exists(config_filepath) is False:
    raise FileNotFoundError(f"Config file not found at {config_filepath}")
config = download.read_config(config_filepath)
# set the API key in the environment and store it
if config.get("DEFAULT","API_KEY") == "":
    raise ValueError("API_KEY not found in config file. Please enter your API key in the config file and try again")
os.environ["API_KEY"] = config["DEFAULT"]["API_KEY"]
auth = Auth.from_env("API_KEY")
auth.store()

# Create the output path that the order will be saved to
output_path = os.path.join(os.getcwd(), "downloads", order_name)
print(f"Order will be saved to {output_path}")


# Download the order or if the order already exists, get the existing order
# if you want to force a download,reguardless if an order with the same name exists or not set overwrite=True
asyncio.run(
    download.download_order_by_name(
        order_name,
        output_path,
        roi,
        start_date,
        end_date,
        overwrite=False,            # if True will overwrite an existing order with the same name and download the new one
        continue_existing=continue_existing,    # if True will continue downloading an existing order with the same name
        cloud_cover=CLOUD_COVER,
        product_bundle="analytic_udm2",
        coregister=False,           # if True will coregister the images to the image with the lowest cloud cover using Planet's coregistration service
        min_area_percentage=min_area_percentage,    # minimum area percentage of the ROI's area that must be covered by the images to be downloaded
        month_filter=month_filter # filter by months
    )
)

if os.path.exists(output_path):
    print(f"Order saved to {output_path}")
    print(f"Ready for processing")
