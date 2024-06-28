import os
import shutil
import numpy as np
import rasterio
from rasterio.plot import show
from coastseg_planet import processing
from coastseg_planet.processing import get_best_landsat_from_dir, get_best_landsat_tifs
from coastseg_planet import masking
import glob
import os
from coastseg_planet.processing import get_tiffs_with_bad_area
from coastseg_planet import download
from coastseg_planet import model
from coastseg_planet.masking import apply_cloudmask_to_dir
from coastseg_planet import coregister
from coastseg_planet import visualize

# This script is used to preprocess all the landsat and planet images for the CoastSeg-Planet model
# it creates the landsat cloud mask, applies the cloud mask to the planet images, and then coregisters the images

# you need to make sure the landsat and planet images are in the same CRS before coregistering
# always attempt to reproject the landsat image before attempt to reprojection the planet images

# make sure the cloud mask are in the same CRS as the model_format tiffs
# same goes for the landsat cloud mask


def generate_coreg_movie(
    input_dir:str,
    ref_path:str,
    ref_mask:str,
    base_movie_name:str,
    output_suffix="_TOAR_processed_coregistered_global.tif",
    bad_mask_suffix:str = "udm2_clip_combined_mask.tif",
    use_local:bool=True,
    overwrite:bool=False,
    **kwargs,
):
    """
    Generate a movie of coregistered TIFF files.

    Args:
        input_dir (str): The directory containing the TIFF files.
        ref_path (str): The path to the reference image for coregistration.
        ref_mask (str): The path to the reference mask for coregistration.
        base_movie_name (str): The base name for the output movie file.
        output_suffix (str, optional): The suffix to be added to the coregistered TIFF files. Defaults to "_TOAR_processed_coregistered_global.tif".
        use_local (bool, optional): Flag indicating whether to use local coregistration. Defaults to True.
        overwrite (bool, optional): Flag indicating whether to overwrite existing output movie file. Defaults to False.
    """
    coregister.coregister_directory(
        input_dir,
        ref_path,
        ref_mask,
        output_suffix=output_suffix,
        bad_mask_suffix=bad_mask_suffix,
        use_local=use_local,
        overwrite=overwrite,
        **kwargs,
    )
    # make a movie of the coregistered tiffs
    tiff_files = sorted(glob.glob(os.path.join(input_dir, f"*{output_suffix}")))
    coreg_str = "local" if use_local else "global"
    output_movie = os.path.join(input_dir, f"{base_movie_name}_{coreg_str}.mp4")
    if os.path.exists(output_movie):
        os.remove(output_movie)
    visualize.create_movie_from_tiffs(tiff_files, output_movie)


kwargs = {
    "max_shift": 2,  # maximum shift distance in reference image pixel units (default: 5 px)
    "align_grids": True,
    "tieP_filter_level": 3,
    "min_reliability": 50,
    # "min_reliability": 0.50,
    "grid_res": 100,  # tie point grid resolution in pixels of the target image (x-direction) (the lower this value the longer it takes)
    "max_points": 1000,  # maximum number of tie points to use for coregistration
    "r_b4match": 1,  # band of reference image to be used for matching (starts with 1; default: 1)
    "s_b4match": 1,  # band of source image to be used for matching (starts with 1; default: 1)
    "v": True,  # verbose mode
    "q": False,  # quiet mode
    "ignore_errors": True,  # Recommended to set to True for batch processing
    "progress": True,
    "out_crea_options": [
        "COMPRESS=LZW"
    ],  # otherwise if will use deflate which will not work well with our model
    "fmt_out": "GTIFF",
    "CPUs": coregister.get_cpus() / 2,
}

# If downloaded with TOAR tool applied don't convert to TOAR again (set this to False)
convert_to_TOAR = False
run_good_bad_classification = False  # @todo make this true
APPLY_WATER_MASK = False

bad_mask_suffix = "udm2_clip_combined_mask.tif"

# input the coastseg session with landsat data downloaded
# downloaded_directory = (
#     r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime06-07-24__02_01_53"
# )
# this is where we would download the ROI with planet or load it if it was already downloaded
planet_dir = r"C:\development\coastseg-planet\downloads\Santa_Cruz_boardwalk_TOAR_enabled_analytic_udm2_full_dataset_cloud_cover_60\de351ce5-5797-4972-b1c3-381da78ea4e8\PSScene"
good_dir = os.path.join(planet_dir, "good")
model_path = r"C:\development\coastseg-planet\CoastSeg-Planet\models\best_rgb.h5"
csv_path = os.path.join(planet_dir, "classification_results.csv")
output_folder = planet_dir
path_to_inference_imgs = planet_dir

# get the landsat with the lowest RMSE and the cloud cover mask
# landsat_dir = get_best_landsat_from_dir(downloaded_directory)
# landsat_path, raw_landsat_cloud_mask_path = get_best_landsat_tifs(downloaded_directory)

landsat_path = r"C:\development\coastseg-planet\CoastSeg-Planet\landsat\santa_cruz_boardwalk\ID_1_datetime05-21-24__01_55_42\L8\ms\2024-03-13-18-45-55_L8_ID_1_datetime05-21-24__01_55_42_ms.tif"
raw_landsat_cloud_mask_path = r"C:\development\coastseg-planet\CoastSeg-Planet\landsat\santa_cruz_boardwalk\ID_1_datetime05-21-24__01_55_42\L8\mask\2024-03-13-18-45-55_L8_ID_1_datetime05-21-24__01_55_42_mask.tif"
# print(f"The best satellite is {landsat_dir}")
print(f"The best landsat tiff is at : {landsat_path}")
print(f"The best landsat cloud mask is at : {raw_landsat_cloud_mask_path}")


landsat_processed_path = processing.format_landsat_tiff(landsat_path)
print(
    f"The landsat image that will be used for the model is at : {landsat_processed_path}"
)
landsat_cloud_mask_path = raw_landsat_cloud_mask_path.replace(".tif", "_processed.tif")
masking.save_landsat_cloud_mask(raw_landsat_cloud_mask_path, landsat_cloud_mask_path)

# get the tiffs in the directory
tiff_paths = []
if convert_to_TOAR:
    tiff_paths = glob.glob(os.path.join(planet_dir, "*AnalyticMS_clip.tif"))
else:  # if the TOAR tool was already applied
    tiff_paths = glob.glob(os.path.join(planet_dir, "*3B_AnalyticMS_toar_clip.tif"))
    if len(tiff_paths) == 0:
        print("No TIFFs found in the directory")

print(f"Number of TIFFs: {len(tiff_paths)}")
# @todo need to pass the ROI as well
# get the topobathy data for the site
# download.download_topobathy('AK_roi',save_dir=planet_dir)

# improvement get the planet tiff with the largest area and use that as the expected area
expected_area_km = 4.5  # 4.1 square kilometers
threshold = 0.90
bad_tiff_paths = get_tiffs_with_bad_area(
    tiff_paths, expected_area_km, threshold, use_threads=True
)
print(f"Number of bad TIFFs: {len(bad_tiff_paths)}")
# for each bad tiff move it a new folder called bad
bad_path = os.path.join(planet_dir, "bad")
os.makedirs(bad_path, exist_ok=True)
for bad_tiff in bad_tiff_paths:
    # move the file to the bad folder
    shutil.move(bad_tiff, bad_path)

# convert the files in the directory to TOAR (Top of Atmosphere Reflectance)
if convert_to_TOAR:
    processing.convert_directory_to_TOAR(
        planet_dir,
        input_suffix="AnalyticMS_clip.tif",
        output_suffix="_TOAR.tif",
        separator="_3B",
    )

# convert to RGB format that is compatible with the model. Range of values is 0-255
if convert_to_TOAR:
    input_suffix = "3B_TOAR.tif"
else:
    input_suffix = "_3B_AnalyticMS_toar_clip.tif"

crs = None
# crs = rasterio.open(tiff_paths[0]).crs.to_epsg()
# crs =32604
# processing.reproject_raster_in_place(landsat_processed_path, crs)
# processing.reproject_raster_in_place(landsat_cloud_mask_path, crs)
# only convert if the planet rasters do not have the same CRS as the landsat raster
# if rasterio.open(landsat_processed_path).crs != rasterio.open(tiff_paths[0]).crs:
#     crs = rasterio.open(landsat_processed_path).crs.to_epsg()
#     print(f"Converting to CRS: {crs}")

processing.convert_directory_to_model_format(
    planet_dir,
    input_suffix=input_suffix,
    output_suffix="_TOAR_model_format.tif",
    crs=crs,
    separator="_3B",
)


regex = "*TOAR_model_format"
# use a parameter to control if the good bad classification should run again
if run_good_bad_classification:
    model.run_classification_model(
        model_path,
        path_to_inference_imgs,
        output_folder,
        csv_path,
        regex,
        move_files=False,
    )
# if the classification model was not run then
if not os.path.exists(good_dir):
    good_dir = planet_dir

# apply cloud mask to directory of planet images and generate cloud masks in that directory for each tif
apply_cloudmask_to_dir(good_dir,output_suffix="_combined_mask.tif")

# if the coregistered tiffs already exist delete them
# delete any of the existing coregistered tiffs
tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*coregistered*.tif")))
for tiff_file in tiff_files:
    print(f"Removing {os.path.basename(tiff_file)}")
    os.remove(tiff_file)



if APPLY_WATER_MASK:
    # run the model for each good tiff
    model.apply_model_to_dir(good_dir,"_TOAR_model_format.tif")

    # update the bad mask with the water mask
    # @todo don't have this be hardcoded
    model_card_path = r'C:\development\coastseg-planet\CoastSeg-Planet\output_zoo\coastseg_planet\coastseg_planet\config\model_card.yml'
    masking.create_water_bad_mask_directory(good_dir,model_card_path,
                                            input_suffix="_combined_mask.tif",
                                            output_suffix="_bad_mask.tif",
                                            npz_suffix="_TOAR_model_format_res.npz")
    # set the bad mask suffix to be the updated bad mask that contains the water mask
    bad_mask_suffix="_bad_mask.tif"

# apply global coregistration
base_movie_name = "Santa_Cruz_boardwalk_TOAR_enabled_analytic_udm2_full_dataset_cloud_cover_60"
generate_coreg_movie(
    good_dir,
    landsat_processed_path,
    landsat_cloud_mask_path,
    base_movie_name,
    output_suffix="_TOAR_processed_coregistered_global.tif",
    bad_mask_suffix=bad_mask_suffix,
    use_local=False,
    overwrite=True,
    **kwargs,
)
print(f"Completed global coregistering {len(tiff_paths)} tiffs")

# apply local coregistration
generate_coreg_movie(
    good_dir,
    landsat_processed_path,
    landsat_cloud_mask_path,
    base_movie_name,
    output_suffix="_TOAR_processed_coregistered_local.tif",
    bad_mask_suffix=bad_mask_suffix,
    use_local=True,
    overwrite=True,
    **kwargs,
)
print(f"Completed local coregistering {len(tiff_paths)} tiffs")

# do it again but use the original tiff before coregistering
# specify the pattern to match your tiff files
tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*_3B_TOAR_model_format.tif")))
output_movie = os.path.join(good_dir, f"{base_movie_name}_original.mp4")
if os.path.exists(output_movie):
    os.remove(output_movie)
visualize.create_movie_from_tiffs(tiff_files, output_movie)
