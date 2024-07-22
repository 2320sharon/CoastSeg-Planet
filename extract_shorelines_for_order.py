# Standard Library Imports
import os
import glob

# Third-party Library Imports
import warnings
import geopandas as gpd

# Project-specific Imports
from coastseg_planet import processing
from coastseg_planet import model
from coastseg_planet import coregister
from coastseg_planet.utils import filter_files_by_area
from coastseg_planet import transects
from coastseg_planet import shoreline_extraction

from coastseg import file_utilities

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

download_settings = {}

extract_shorelines_settings = {
    'output_epsg': 4326,       # native epsg of the ROI 
    'min_length_sl': 100,      # minimum length of the shoreline to be considered
    'dist_clouds': 50,         # distance to remove clouds from the shoreline
    'min_beach_area': 1000,    # minimum area of the beach to be considered
    'max_dist_ref': 300,       # maximum distance to the reference shoreline in which shorelines will be extracted. Eg if 300, shorelines will be extracted within 300m of the reference shoreline
    'satname': 'planet',       # Name of the satellite used to capture the images
}

model_settings = {}

# Enter the locations of the feature inputs
# ----------------
# 1. Enter the path ROI = Region of Interest used to download your order
roi_path = os.path.join(os.getcwd(),"sample_data", "rois.geojson")
# 2. Enter the path to the transects file
# transects_path = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\transects.geojson"
transects_path =os.path.join(os.getcwd(),"sample_data", "transects.geojson")
# 3. Enter the path to the reference shoreline file
# shoreline_path = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\good\shoreline.geojson"
shoreline_path =os.path.join(os.getcwd(),"sample_data", "shoreline.geojson")
# ----------------

if not os.path.exists(roi_path):
    raise ValueError('ROI location does not exist. Please provide a valid path to the ROI.')
if not os.path.exists(transects_path):
    raise ValueError('Transects path does not exist. Please provide a path to the transects to intersect with the shorelines.')
if not os.path.exists(shoreline_path):
    raise ValueError('Shoreline path does not exist. Please provide a path to the shoreline to intersect with the shorelines.')

# 4. Enter the location of directory containing the downloaded imagery from Planet
#   - Make sure this directory contains the tif,json,and xml files for the entire order
planet_dir = ''
planet_dir = r"C:\development\coastseg-planet\CoastSeg-Planet\downloads\DUCK_pier_cloud_0.7_TOAR_enabled_2023-06-01_to_2023-08-01\31eea859-d1dd-46c2-b6a4-7c15871be926\PSScene"
good_dir = os.path.join(planet_dir, 'good') # this is where the good imagery will be stored

# Settings
# ----------------
# 5. Set the settings for the extraction of the shorelines
CONVERT_TO_MODEL_FORMAT = True  # Set to False if the files are already in the model format bands Red,Blue,Green values (0-255)
CONVERT_TOAR = False            # If downloaded with TOAR tool(performed by default) was applied don't convert to TOAR again (set this to False) 
RUN_GOOD_BAD_CLASSIFER = True   # Whether to run the classification model or not to sort the images into good and bad directories


# Model Inputs
model_path = r"C:\development\coastseg-planet\CoastSeg-Planet\models\best_rgb.h5"
model_card_path = r'C:\development\coastseg-planet\CoastSeg-Planet\output_zoo\coastseg_planet\coastseg_planet\config\model_card.yml'
weights_directory = r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\downloaded_models\segformer_RGB_4class_8190958'
model_card_path = file_utilities.find_file_by_regex(
    weights_directory, r".*modelcard\.json$"
)

# Filter out the files that are less than 90% of the ROI area
filter_files_by_area(planet_dir,threshold=0.90,roi_path=roi_path,verbose=True)

# convert the files in the directory to TOAR (Top of Atmosphere Reflectance) 
if CONVERT_TOAR:
    processing.convert_directory_to_TOAR(planet_dir,input_suffix='AnalyticMS_clip.tif',output_suffix='_TOAR.tif',separator='_3B')

# convert to RGB format that is compatible with the model. Range of values is 0-255
if CONVERT_TOAR:
    input_suffix='3B_TOAR.tif'
else:
    input_suffix='_3B_AnalyticMS_toar_clip.tif'

if CONVERT_TO_MODEL_FORMAT:
    processing.convert_directory_to_model_format(planet_dir,input_suffix=input_suffix,output_suffix='_TOAR_model_format.tif',separator='_3B')


# use a parameter to control if the good bad classification should run again
# set move files to be true so that the associated files like the cloud mask & xml are moved to the good directory
if RUN_GOOD_BAD_CLASSIFER:
    model.run_classification_model(model_path, planet_dir, planet_dir, regex_pattern= '*TOAR_model_format', move_files=True)
# if the classification model was not run then 
if not os.path.exists(good_dir):
    good_dir = planet_dir

shoreline_gdf = gpd.read_file(shoreline_path)
out_epsg = shoreline_gdf.estimate_utm_crs().to_epsg()
extract_shorelines_settings['output_epsg'] = out_epsg

ref_sl = shoreline_extraction.get_reference_shoreline_as_array(shoreline_gdf,out_epsg)

# suffix of the tif files to extract shorelines from
separator = '_3B'
suffix = f"{separator}_TOAR_model_format"

filtered_tiffs = glob.glob(os.path.join(good_dir, f"*{suffix}.tif"))
if len(filtered_tiffs) == 0:
    print("No tiffs found in the directory")

# then intersect these shorelines with the transects
shorelines_dict = shoreline_extraction.extract_shorelines_with_reference_shoreline(good_dir,
                                                          suffix,
                                                          model_card_path,
                                                          ref_sl,
                                                        extract_shorelines_settings,
                                                        )

# INTERSECT SHORELINES WITH TRANSECTS
if not os.path.exists(transects_path):
    raise ValueError('Transects path does not exist. Please provide a path to the transects to intersect with the shorelines.')
transects.intersect_transects(transects_path,shorelines_dict,extract_shorelines_settings['output_epsg'], save_location=good_dir)