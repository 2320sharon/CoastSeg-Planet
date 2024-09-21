# Standard Library Imports
import os
import zipfile
import glob
import logging
import warnings
from osgeo import gdal
gdal.UseExceptions()

# Suppress specific warning messages
warnings.filterwarnings('ignore', message='Some layers from the model checkpoint')

# Third-party Library Imports
import geopandas as gpd

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# Set TensorFlow logging level to suppress informational and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # '0' = all messages are logged (default behavior), '1' = INFO messages are not printed, '2' = INFO and WARNING messages are not printed, '3' = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Project-specific Imports
from coastseg_planet import processing
from coastseg_planet import model
from coastseg_planet import utils
from coastseg_planet.utils import filter_files_by_area
from coastseg_planet import transects
from coastseg_planet import shoreline_extraction

download_settings = {}

extract_shorelines_settings = {
    'output_epsg': 4326,       # Enter the native epsg of the ROI 
    'min_length_sl': 100,      # minimum length of the shoreline to be considered
    'dist_clouds': 50,         # distance to remove clouds from the shoreline
    'min_beach_area': 1000,    # minimum area of the beach to be considered
    'max_dist_ref': 400,       # maximum distance to the reference shoreline in which shorelines will be extracted. Eg if 300, shorelines will be extracted within 300m of the reference shoreline
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
print(f"transects_path: {transects_path}")
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
#   - If you are using your own data make sure you have it open to the PSScene directory that contains the tif files
# planet_dir = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\DUCK_pier_cloud_0.7_TOAR_enabled_2023-06-01_to_2023-07-01\5576432c-cc59-49e6-882b-3b6ee3365c11\PSScene"
# if you are using your own data make sure you have it open to the PSScene directory that contains the tif files

# FOR NOW WE ARE GOING TO USE THE SAMPLE DATA INCLUDED IN A ZIP FILE
# unzip the sample data 
sample_zip =os.path.join(os.getcwd(),"sample_data", "sample_tiffs.zip")
if os.path.exists(sample_zip):
    with zipfile.ZipFile(sample_zip, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(os.getcwd(),"sample_data", ))

# alternatively you can use the path below to unzip the sample data
planet_dir = sample_zip =os.path.join(os.getcwd(),"sample_data", "sample_tiffs")
good_dir = os.path.join(planet_dir, 'good') # this is where the good imagery will be stored

# Settings
# ----------------
# 5. Set the settings for the extraction of the shorelines
CONVERT_TO_MODEL_FORMAT = True  # Set to False if the files are already in the model format bands Red,Blue,Green values (0-255)
CONVERT_TOAR = False            # If downloaded with TOAR tool(performed by default) was applied don't convert to TOAR again (set this to False) 
RUN_GOOD_BAD_CLASSIFER = True   # Whether to run the classification model or not to sort the images into good and bad directories


# Model Inputs
MODEL_NAME = 'segformer_RGB_4class_8190958'
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
    model.sort_imagery(planet_dir, planet_dir, regex_pattern= '*TOAR_model_format', move_files=True)
# if the classification model was not run then  set the good directory to the planet directory containing the tif files
if not os.path.exists(good_dir):
    good_dir = planet_dir

shoreline_gdf = gpd.read_file(shoreline_path)
out_epsg = shoreline_gdf.estimate_utm_crs().to_epsg()
extract_shorelines_settings['output_epsg'] = out_epsg

# suffix of the tif files to extract shorelines from
separator = '_3B'
suffix = f"{separator}_TOAR_model_format"

filtered_tiffs = glob.glob(os.path.join(good_dir, f"*{suffix}.tif"))
if len(filtered_tiffs) == 0:
    print("No tiffs found in the directory")


# then intersect these shorelines with the transects (comment this line out if you are loading from a file)
shorelines_dict = shoreline_extraction.extract_shorelines_with_reference_shoreline_gdf(good_dir,
                                                          suffix,
                                                          reference_shoreline=shoreline_gdf,
                                                        extract_shorelines_settings = extract_shorelines_settings,
                                                        model_name = MODEL_NAME,
                                                        )


# save the shoreline dictionary to a json file
shoreline_dictionary_path = os.path.join(good_dir,'shorelines_dict.json')

# commment the line below out if you are loading the shoreline dictionary from a json file
utils.save_to_json(shorelines_dict,shoreline_dictionary_path)

# optionally load the shoreline dictionary from a json file if you have already run the extraction and want to skip the extraction step. Uncomment the lines below
# if os.path.exists(shoreline_dictionary_path):
#     shorelines_dict = utils.load_data_from_json(shoreline_dictionary_path)

# INTERSECT SHORELINES WITH TRANSECTS
if not os.path.exists(transects_path):
    raise ValueError('Transects path does not exist. Please provide a path to the transects to intersect with the shorelines.')
transects.intersect_transects(transects_path,shorelines_dict,extract_shorelines_settings['output_epsg'], save_location=good_dir)