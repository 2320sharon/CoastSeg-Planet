# Standard Library Imports
import os
import glob

# Third-party Library Imports
import warnings
import geopandas as gpd

# Project-specific Imports
from coastseg_planet import processing
from coastseg_planet import download
from coastseg_planet import model
from coastseg_planet import coregister
from coastseg_planet.download import download_topobathy
from coastseg_planet.masking import apply_cloudmask_to_dir
from coastseg_planet.utils import filter_files_by_area
from coastseg_planet import transects
from coastseg_planet import shoreline_extraction
from coastseg_planet.processing import create_elevation_mask_utm

from coastseg import file_utilities

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Inputs
#--------------------------------------------------------------------------

download_settings = {}

extract_shorelines_settings = {
    'output_epsg': 4326,
    'min_length_sl': 100,
    'dist_clouds': 50,
    'min_beach_area': 1000,
    'satname': 'planet',
}

coregister_settings = {
   'max_shift':2,
   'align_grids':True,
    "tieP_filter_level": 3,
    "min_reliability": 50,
    "grid_res": 100,  # tie point grid resolution in pixels of the target image (x-direction) (the lower this value the longer it takes)
   'v':True, #verbose mode
   'q':False, #quiet mode
    "r_b4match": 1,  # band of reference image to be used for matching (starts with 1; default: 1)
    "s_b4match": 1,  # band of source image to be used for matching (starts with 1; default: 1)
   'ignore_errors':False, # Recommended to set to True for batch processing
    'progress':True,
    'out_crea_options':["COMPRESS=LZW"], # otherwise if will use deflate which will not work well with our model
    'fmt_out':'GTIFF',
    'CPUs' : coregister.get_cpus()/2,
}
model_settings = {}

# Inputs
# ----------------
# ROI = Region of Interest
ROI_location = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\roi.geojson"
transects_path = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\transects.geojson" # OPTIONAL   LOCATION OF TRANSECTS
sitename = 'AK'
# location of directory containing the downloaded imagery from Planet
planet_dir = ''
planet_dir = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene"
good_dir = os.path.join(planet_dir, 'good')
# Optional Inputs
#----------------
# location of the file containing the reference landsat or planet image to coregister to
reference_path = ''
# location of the cloud mask of the reference landsat or planet image to coregister to
reference_bad_mask_path = ''

# Model Inputs
model_path = r"C:\development\coastseg-planet\CoastSeg-Planet\models\best_rgb.h5"
model_card_path = r'C:\development\coastseg-planet\CoastSeg-Planet\output_zoo\coastseg_planet\coastseg_planet\config\model_card.yml'
weights_directory = r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\downloaded_models\segformer_RGB_4class_8190958'
model_card_path = file_utilities.find_file_by_regex(
    weights_directory, r".*modelcard\.json$"
)
#--------------------------------------------------------------------------

# Controls
CONVERT_TOAR = False # If downloaded with TOAR tool was applied don't convert to TOAR again (set this to False)
RUN_GOOD_BAD_CLASSIFER = False  # Whether to run the classification model or not to sort the images into good and bad directories
APPLY_COREGISTER = False  # Whether to apply coregistration or not


# Filter out the files whose area is too small
filter_files_by_area(planet_dir,threshold=0.90,roi_path=ROI_location,verbose=True)

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
    model.run_classification_model(model_path, planet_dir, planet_dir, regex= '*TOAR_model_format', move_files=False)
# if the classification model was not run then 
if not os.path.exists(good_dir):
    good_dir = planet_dir


if APPLY_COREGISTER: 
    if not os.path.exists(reference_path):
        raise ValueError('Reference path to coregister all the imagery to does not exist. Please provide a reference path to coregister to.')
    if not os.path.exists(reference_bad_mask_path):
        raise ValueError('Reference bad mask path to coregister all the imagery to does not exist. Please provide a reference path to coregister to.')
    coregister.coregister_directory(good_dir,
                                    reference_path,
                                    reference_bad_mask_path,
                                    output_suffix='_TOAR_processed_coregistered_local.tif',
                                    bad_mask_suffix='_combined_mask.tif',
                                    use_local=True,  # if True, use local coregistration, otherwise use global coregistration
                                    overwrite=False, # if True, will overwrite the existing coregistered files, otherwise they will be skipped
                                    **coregister_settings)
    
# Create a shoreline buffer from the topobathy data
roi_gdf = gpd.read_file(ROI_location)
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
if APPLY_COREGISTER:
    suffix = f"_3B_TOAR_processed_coregistered_global"

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