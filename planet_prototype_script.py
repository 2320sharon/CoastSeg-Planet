# Standard Library Imports
import os
import glob

# Third-party Library Imports
import warnings
import geopandas as gpd

# Project-specific Imports
from coastseg_planet import processing
from coastseg_planet import model
from coastseg_planet.download import download_topobathy
from coastseg_planet.utils import filter_files_by_area
from coastseg_planet import transects
from coastseg_planet import shoreline_extraction
from coastseg_planet.processing import create_elevation_mask_utm

from coastseg import file_utilities

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# os.environ['LINE_PROFILE'] = '1'

# Inputs
#--------------------------------------------------------------------------

# This script extracts shoreline using an elevation mask as the reference shoreline buffer to extract shorelines from imagery.
# The script also intersects the extracted shorelines with the transects to get the intersection points.

# Users need to run pip install bathyreq before running this script

download_settings = {}

extract_shorelines_settings = {
    'output_epsg': 4326,
    'min_length_sl': 100,
    'dist_clouds': 50,
    'min_beach_area': 1000,
    'satname': 'planet',
}


model_settings = {}

# Inputs
# ----------------
# ROI = Region of Interest
roi_path = os.path.join(os.getcwd(),"sample_data", "rois.geojson")
transects_path =os.path.join(os.getcwd(),"sample_data", "transects.geojson")

sitename = 'AK'
# location of directory containing the downloaded imagery from Planet
planet_dir = ''
planet_dir = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\DUCK_pier_cloud_0.7_TOAR_enabled_2023-06-01_to_2023-07-01\5576432c-cc59-49e6-882b-3b6ee3365c11\PSScene"
good_dir = os.path.join(planet_dir, 'good')
# Optional Inputs
#----------------
# location of the file containing the reference landsat or planet image to coregister to
reference_path = ''
# location of the cloud mask of the reference landsat or planet image to coregister to
reference_bad_mask_path = ''

# Model Inputs
from coastseg_planet import model
weights_directory = model.get_model_location('segformer_RGB_4class_8190958')
# weights_directory = r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\downloaded_models\segformer_RGB_4class_8190958'
model_card_path = file_utilities.find_file_by_regex(
    weights_directory, r".*modelcard\.json$"
)
#--------------------------------------------------------------------------

# Controls
CONVERT_TOAR = False # If downloaded with TOAR tool was applied don't convert to TOAR again (set this to False)
RUN_GOOD_BAD_CLASSIFER = False  # Whether to run the classification model or not to sort the images into good and bad directories


# Filter out the files whose area is too small
filter_files_by_area(planet_dir,threshold=0.90,roi_path=roi_path,verbose=True)

# convert the files in the directory to TOAR (Top of Atmosphere Reflectance) 
if CONVERT_TOAR:
    processing.convert_directory_to_TOAR(planet_dir,input_suffix='AnalyticMS_clip.tif',output_suffix='_TOAR.tif',separator='_3B')

# convert to RGB format that is compatible with the model. Range of values is 0-255
if CONVERT_TOAR:
    input_suffix='3B_TOAR.tif'
else:
    input_suffix='_3B_AnalyticMS_toar_clip.tif'
    processing.convert_directory_to_model_format(planet_dir,input_suffix=input_suffix,output_suffix='_TOAR_model_format.tif',separator='_3B')

# use a parameter to control if the good bad classification should run again
if RUN_GOOD_BAD_CLASSIFER:
    model.run_classification_model( planet_dir, planet_dir, regex= '*TOAR_model_format', move_files=False)
# if the classification model was not run then 
if not os.path.exists(good_dir):
    good_dir = planet_dir

    
# Create a shoreline buffer from the topobathy data
roi_gdf = gpd.read_file(roi_path)
topobathy_tiff = download_topobathy(sitename,roi_gdf,planet_dir)

# These examples inputs are for Unakeleet, Alaska
low_threshold = -1.57
high_threshold = 3.527516937255859
masked_elevation_tif_path = create_elevation_mask_utm(topobathy_tiff,low_threshold,high_threshold)
print(f"Masked elevation tif created at {masked_elevation_tif_path}")
# EXRACT SHORELINES
#-------------------

# get the output epsg from the roi
out_epsg = roi_gdf.estimate_utm_crs().to_epsg()
extract_shorelines_settings['output_epsg'] = out_epsg

# suffix of the tif files to extract shorelines from
suffix = f"_3B_TOAR_model_format"


filtered_tiffs = glob.glob(os.path.join(good_dir, f"*{suffix}.tif"))
if len(filtered_tiffs) == 0:
    print("No tiffs found in the directory")

# then intersect these shorelines with the transects
shorelines_dict = shoreline_extraction.extract_shorelines(good_dir,
                                                          suffix,
                                                          model_card_path,
                                                        extract_shorelines_settings,
                                                        masked_elevation_tif_path,
                                                        cloud_mask_suffix='3B_udm2_clip.tif')

# INTERSECT SHORELINES WITH TRANSECTS
if not os.path.exists(transects_path):
    raise ValueError('Transects path does not exist. Please provide a path to the transects to intersect with the shorelines.')
transects.intersect_transects(transects_path,shorelines_dict,extract_shorelines_settings['output_epsg'], save_location=good_dir)