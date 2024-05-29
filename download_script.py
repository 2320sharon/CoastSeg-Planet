from coastseg_planet import download
from planet import Auth
import os
import asyncio
import json

# 1. name the order
# Either enter the name of an existing order or create a new one
order_name = 'santa_cruz_boardwalk_TOAR_enabled4'

# 2. insert path to roi geojson
roi_path = r"C:\development\coastseg-planet\CoastSeg-Planet\boardwalk\roi.geojson"
with open(roi_path, 'r') as file:
    roi = json.load(file)
    
# 3. Select a start and end date YYYY-MM-DD
start_date = '2023-04-01'
end_date = '2023-04-15'
    
# 4. read the api key from the config file and set it in the environment
# if one doesnt exist, create a config file with the following format
# [DEFAULT]
# API_KEY = <PLANET API KEY>

config_filepath =r"C:\development\coastseg-planet\CoastSeg-Planet\config.ini"
config = download.read_config(config_filepath)
# set the API key in the environment and store it
os.environ['API_KEY'] = config['DEFAULT']['API_KEY']    
auth = Auth.from_env('API_KEY')
auth.store()

# Create the output path that the order will be saved to
output_path = os.path.join(os.getcwd(), 'downloads', order_name)
print(f"Order will be saved to {output_path}")
# Download the order or if the order already exists, get the existing order
# if you want to force a download,reguardless if an order with the same name exists or not set overwrite=True
asyncio.run(download.download_order_by_name(order_name,output_path,roi,start_date,end_date,overwrite=False))

if os.path.exists(output_path):
    print(f"Order saved to {output_path}")
    print(f"Ready for processing")
