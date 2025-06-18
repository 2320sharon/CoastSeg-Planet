# Standard library imports
from shutil import copyfile
from typing import List, Optional, Union, Tuple
from xml.dom import minidom
import concurrent.futures
from shapely.geometry import mapping
from rasterio.mask import mask
from datetime import datetime, timezone
import glob
import os
import shutil
import json
from collections import Counter
import tqdm
import logging
from shapely.geometry import shape, mapping


# Third party imports
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal, osr
from skimage import img_as_ubyte
from skimage.io import imsave
import geopandas as gpd
import numpy as np
import rasterio
import shapely.geometry as geometry
import skimage.morphology as morphology
import pyproj

from coastseg_planet import db_utils


def process_analytic_tile(
    analytic_path, xml_path, roi_path, output_folder
) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        output_path = create_new_output_path(
            analytic_path, suffix="_clip_TOAR", output_location=output_folder
        )
        clip_toa_reproject_geotiff(analytic_path, xml_path, roi_path, output_path)
        return "success", output_path, None
    except Exception as e:
        return "failed", None, str(e)


def clip_geotiff_to_roi(
    geotiff_path: str, roi_gdf: gpd.GeoDataFrame, output_path: Optional[str] = None
) -> str:
    """
    Clips a GeoTIFF raster to the extent of a given ROI (Region of Interest) geometry and generates a new GeoTIFF file.
    Parameters:
        geotiff_path (str): Path to the input GeoTIFF file.
        roi_gdf (gpd.GeoDataFrame): GeoDataFrame containing the ROI geometry. The first row's geometry is used for clipping.
        output_path (Optional[str], optional): Path to save the clipped GeoTIFF. If None, appends '_clip.tif' to the input filename.
    Returns:
        str: Path to the saved clipped GeoTIFF file.
    Raises:
        ValueError: If the ROI does not intersect with the GeoTIFF bounds.
    Notes:
        - The function reprojects the ROI geometry to match the CRS of the input raster before clipping.
        - The output GeoTIFF is compressed using LZW compression.
    """
    with rasterio.open(geotiff_path) as src:
        src_crs = src.crs
        roi_gdf = roi_gdf.to_crs(src_crs)
        roi_geom = roi_gdf.iloc[0]["geometry"]

        # Check if ROI intersects with raster bounds
        raster_bounds_geom = shape(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [src.bounds.left, src.bounds.bottom],
                        [src.bounds.left, src.bounds.top],
                        [src.bounds.right, src.bounds.top],
                        [src.bounds.right, src.bounds.bottom],
                        [src.bounds.left, src.bounds.bottom],
                    ]
                ],
            }
        )
        if not roi_geom.intersects(raster_bounds_geom):
            raise ValueError("ROI does not intersect with the GeoTIFF bounds.")

        out_image, out_transform = mask(src, [mapping(roi_geom)], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )

        if output_path is None:
            base, _ = os.path.splitext(geotiff_path)
            output_path = f"{base}_clip.tif"

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    return output_path


def process_udm_tile(
    udm_path, roi_path, output_folder
) -> Tuple[str, Optional[str], Optional[str]]:
    if not udm_path:
        return "missing", None, None
    try:
        roi_gdf = gpd.read_file(roi_path)
        output_path = create_new_output_path(
            udm_path, suffix="_clip", output_location=output_folder
        )
        clip_geotiff_to_roi(udm_path, roi_gdf, output_path)
        return "success", output_path, None
    except Exception as e:
        return "failed", None, str(e)


def process_tile(
    tile_id: str,
    files: list,
    roi_path: str,
    output_folder: str,
    require_analytic_success_for_udm: bool = True,
) -> dict:
    log = {
        "tile_id": tile_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analytic_status": None,
        "udm_status": None,
        "analytic_path": None,
        "udm_path": None,
        "xml_path": None,
        "json_path": None,
        "errors": [],
    }

    paths = find_tile_files(files)

    if paths["xml"] and paths["analytic"]:
        analytic_status, analytic_path, analytic_err = process_analytic_tile(
            paths["analytic"], paths["xml"], roi_path, output_folder
        )
        log["analytic_status"] = analytic_status
        log["analytic_path"] = analytic_path
        if analytic_err:
            log["errors"].append(f"Analytic error: {analytic_err}")
    else:
        log["analytic_status"] = "missing"
        if not paths["xml"]:
            log["errors"].append("Missing XML file.")
        if not paths["analytic"]:
            log["errors"].append("Missing Analytic TIFF.")

    if require_analytic_success_for_udm and log["analytic_status"] != "success":
        log["udm_status"] = "skipped"
    else:
        udm_status, udm_path, udm_err = process_udm_tile(
            paths["udm"], roi_path, output_folder
        )
        log["udm_status"] = udm_status
        log["udm_path"] = udm_path
        if udm_err:
            log["errors"].append(f"UDM error: {udm_err}")

    # Save XML and JSON metadata files to output folder
    copied_xml_path = copy_file_to_output(paths["xml"], output_folder)

    copied_json_path = copy_file_to_output(paths["json"], output_folder)

    log["xml_path"] = copied_xml_path
    log["json_path"] = copied_json_path

    return log


def run_pipeline(
    roi_path: str,
    output_folder: str,
    date_range: Tuple[str, str],
    min_overlap: float,
    db_path: str,
    log_path: Optional[str] = None,
    require_analytic_success_for_udm: bool = True,
    quick_fail: bool = False,
) -> List[dict]:
    """
    Main processing loop. This queries the database for tiles that intersect the ROI,
    clips and converts them to TOA, and writes logs + summaries.

    Parameters:
    - require_analytic_success_for_udm: If True, UDM processing is skipped if Analytic fails.
    - quick_fail: If True, stops after the first error.
    - log_path: If provided, writes a JSONL log of processing results.
    Returns:
    - List of log entries for each tile processed.
    """
    os.makedirs(output_folder, exist_ok=True)
    processor = db_utils.create_processor(db_path)
    # Load ROI geometry from the provided path as a dictionary so we can query the database with it
    roi_geom = load_roi_geometry_as_geojson(roi_path)
    tiles = query_tiles(roi_geom, date_range, min_overlap, processor)

    log_entries = []

    for tile_id, files in tqdm.tqdm(tiles.items(), desc="Processing Tiles"):
        log_entry = process_tile(
            tile_id,
            files,
            roi_path,
            output_folder,
            require_analytic_success_for_udm=require_analytic_success_for_udm,
        )
        log_entries.append(log_entry)
        print_tile_status(log_entry)

        if quick_fail and (
            log_entry["analytic_status"] == "failed"
            or log_entry["udm_status"] == "failed"
        ):
            print("\n⚠️ Quick fail activated. Halting pipeline due to error.")
            print(f"Tile ID       : {log_entry['tile_id']}")
            print(f"Analytic Stat : {log_entry['analytic_status']}")
            print(f"UDM Stat      : {log_entry['udm_status']}")
            for i, err in enumerate(log_entry["errors"], 1):
                print(f"Error {i}      : {err}")
            break

    if log_path:
        write_log(log_path, log_entries)
        print(f"\n📄 Log written to: {log_path}")

    summarize_log(log_entries)
    return log_entries


def find_tile_files(files: list) -> dict:
    """
    Finds specific tile files from a list of filenames.

    Returns a dictionary with:
    - 'xml'       : metadata XML file (e.g. endswith 'metadata.xml')
    - 'json'      : tile metadata JSON file (filename contains 'metadata.json')
    - 'udm'       : UDM2 file (endswith 'udm2.tif')
    - 'analytic'  : AnalyticMS image (endswith 'AnalyticMS.tif' and not a UDM)
    """
    return {
        "xml": next((f for f in files if f.endswith("metadata.xml")), None),
        "json": next((f for f in files if "metadata.json" in f), None),
        "udm": next((f for f in files if f.endswith("udm2.tif")), None),
        "analytic": next(
            (f for f in files if f.endswith("AnalyticMS.tif") and "udm" not in f), None
        ),
    }


def query_tiles(
    roi_geom: dict, date_range: Tuple[str, str], min_overlap: float, processor
) -> dict:
    query = {
        "geometry": roi_geom,
        "start_date": date_range[0],
        "end_date": date_range[1],
        "min_overlap": min_overlap,
    }
    return processor.query_tiles_table(query=query)


def write_log(log_path: str, entries: List[dict]):
    with open(log_path, "w") as f:
        for entry in entries:
            json.dump(entry, f)
            f.write("\n")


def summarize_log(log_entries: List[dict]):
    a_stats = Counter(entry["analytic_status"] for entry in log_entries)
    u_stats = Counter(entry["udm_status"] for entry in log_entries)
    failed = [e for e in log_entries if e["errors"]]

    print("\n🔍 Pipeline Summary")
    print("  ✅ Analytic Status:", dict(a_stats))
    print("  🧭 UDM Status     :", dict(u_stats))
    print(f"  ❌ Failed Tiles   : {len(failed)}")


def print_tile_status(log_entry):
    a = log_entry["analytic_status"]
    u = log_entry["udm_status"]
    print(
        f"[{log_entry['tile_id']}] Analytic: {a} | UDM: {u} | Errors: {len(log_entry['errors'])}"
    )


def load_roi_geometry_as_geojson(roi_path: str) -> dict:
    """
    Loads the geometry of the first feature from a GeoJSON file containing ROI (Region of Interest) data as a dictionary.

    Args:
        roi_path (str): The file path to the GeoJSON file containing ROI features.

    Returns:
        dict: The geometry dictionary of the first feature in the GeoJSON file.
        Example:
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [-123.456, 45.678],
                    [-123.456, 46.678],
                    [-122.456, 46.678],
                    [-122.456, 45.678],
                    [-123.456, 45.678]
                ]
            ]
        }

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        KeyError: If the expected keys ('features' or 'geometry') are missing in the GeoJSON structure.
    """
    with open(roi_path, "r") as f:
        geojson = json.load(f)
    return geojson["features"][0]["geometry"]


def create_new_output_path(
    input_path: str, suffix: str = "_clip_TOAR", output_location: Optional[str] = None
) -> str:
    base, ext = os.path.splitext(input_path)
    new_path = f"{base}{suffix}{ext}"
    if output_location:
        new_path = os.path.join(output_location, os.path.basename(new_path))
    return new_path


def copy_file_to_output(
    source_path: Optional[str], output_folder: str
) -> Optional[str]:
    """
    Copies a file to the output folder if the path is valid.

    Args:
        source_path (str): Original file path.
        output_folder (str): Destination folder.

    Returns:
        str or None: Destination path if copied, else None.
    """
    if not source_path:
        return None
    try:
        dest_path = os.path.join(output_folder, os.path.basename(source_path))
        os.makedirs(output_folder, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception as e:
        return None


def parse_reflectance_coeffs(xml_path: str) -> dict:
    xmldoc = minidom.parse(xml_path)
    coeffs = {}
    for node in xmldoc.getElementsByTagName("ps:bandSpecificMetadata"):
        bn_node = node.getElementsByTagName("ps:bandNumber")[0]
        coeff_node = node.getElementsByTagName("ps:reflectanceCoefficient")[0]
        if bn_node and coeff_node:
            band = int(bn_node.firstChild.data)
            coeff = float(coeff_node.firstChild.data)
            coeffs[band] = coeff
    return coeffs


def clip_toa_reproject_geotiff(
    image_path: str,
    xml_path: str,
    roi_path: str,
    output_path: str,
    output_epsg: Optional[str] = None,
) -> str:
    BAND_COUNT = 4
    SCALE_FACTOR = 10000

    # Read ROI and align CRS
    roi_gdf = gpd.read_file(roi_path)
    with rasterio.open(image_path) as src:
        if roi_gdf.crs != src.crs:
            roi_gdf = roi_gdf.to_crs(src.crs)

        roi_geom = [mapping(roi_gdf.iloc[0].geometry)]
        clipped, transform = mask(src, roi_geom, crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
                "compress": "lzw",
                "count": BAND_COUNT,
                "dtype": rasterio.uint16,
            }
        )
        src_crs = src.crs

    # Parse reflectance coefficients
    coeffs = parse_reflectance_coeffs(xml_path)

    # Apply TOA conversion
    toa_array = np.array(
        [clipped[i] * coeffs.get(i + 1, 1.0) * SCALE_FACTOR for i in range(BAND_COUNT)]
    ).astype(rasterio.uint16)

    # Optional reprojection
    if output_epsg:
        dst_crs = rasterio.crs.CRS.from_string(output_epsg)
        transform, width, height = calculate_default_transform(
            src_crs,
            dst_crs,
            meta["width"],
            meta["height"],
            *rasterio.transform.array_bounds(
                meta["height"], meta["width"], meta["transform"]
            ),
        )
        reprojected = np.empty((BAND_COUNT, height, width), dtype=rasterio.uint16)
        for i in range(BAND_COUNT):
            reproject(
                source=toa_array[i],
                destination=reprojected[i],
                src_transform=meta["transform"],
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        toa_array = reprojected
        meta.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
    else:
        meta.update({"crs": src_crs})

    # Write final output
    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(BAND_COUNT):
            dst.write(toa_array[i], i + 1)

    print(f"TOA reflectance file saved to: {output_path}")

    return output_path


def process_analytic_tile(
    analytic_path, xml_path, roi_path, output_folder
) -> Tuple[str, Optional[str], Optional[str]]:
    try:
        output_path = create_new_output_path(
            analytic_path, suffix="_clip_TOAR", output_location=output_folder
        )
        clip_toa_reproject_geotiff(analytic_path, xml_path, roi_path, output_path)
        return "success", output_path, None
    except Exception as e:
        return "failed", None, str(e)


def process_udm_tile(
    udm_path, roi_path, output_folder
) -> Tuple[str, Optional[str], Optional[str]]:
    if not udm_path:
        return "missing", None, None
    try:
        roi_gdf = gpd.read_file(roi_path)
        output_path = create_new_output_path(
            udm_path, suffix="_clip", output_location=output_folder
        )
        clip_geotiff_to_roi(udm_path, roi_gdf, output_path)
        return "success", output_path, None
    except Exception as e:
        return "failed", None, str(e)


def process_tile(
    tile_id: str,
    files: list,
    roi_path: str,
    output_folder: str,
    require_analytic_success_for_udm: bool = True,
) -> dict:
    log = {
        "tile_id": tile_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analytic_status": None,
        "udm_status": None,
        "analytic_path": None,
        "udm_path": None,
        "xml_path": None,
        "json_path": None,
        "errors": [],
    }

    paths = find_tile_files(files)

    if paths["xml"] and paths["analytic"]:
        analytic_status, analytic_path, analytic_err = process_analytic_tile(
            paths["analytic"], paths["xml"], roi_path, output_folder
        )
        log["analytic_status"] = analytic_status
        log["analytic_path"] = analytic_path
        if analytic_err:
            log["errors"].append(f"Analytic error: {analytic_err}")
    else:
        log["analytic_status"] = "missing"
        if not paths["xml"]:
            log["errors"].append("Missing XML file.")
        if not paths["analytic"]:
            log["errors"].append("Missing Analytic TIFF.")

    if require_analytic_success_for_udm and log["analytic_status"] != "success":
        log["udm_status"] = "skipped"
    else:
        udm_status, udm_path, udm_err = process_udm_tile(
            paths["udm"], roi_path, output_folder
        )
        log["udm_status"] = udm_status
        log["udm_path"] = udm_path
        if udm_err:
            log["errors"].append(f"UDM error: {udm_err}")

    # Save XML and JSON metadata files to output folder
    copied_xml_path = copy_file_to_output(paths["xml"], output_folder)

    copied_json_path = copy_file_to_output(paths["json"], output_folder)

    log["xml_path"] = copied_xml_path
    log["json_path"] = copied_json_path

    return log


def create_elevation_mask_utm(tiff_file, low_threshold, high_threshold):
    masked_tif_path = create_elevation_mask(
        tiff_file, tiff_file.replace(".tif", "_mask.tif"), low_threshold, high_threshold
    )
    masked_projected_topobathy_tiff = reproject_to_utm(
        masked_tif_path, masked_tif_path.replace(".tif", "_utm.tif")
    )
    return masked_projected_topobathy_tiff


def get_mask_in_matching_projection(source_tiff: str, mask_tiff: str) -> np.ndarray:
    """
    Reprojects the mask TIFF to match the resolution and grid of the source TIFF.
    This returns the mask as a boolean numpy array.

    Parameters:
        source_tiff (str): The path to the source TIFF file.
        mask_tiff (str): The path to the mask TIFF file.

    Returns:
        np.ndarray: The reprojected mask as a numpy array.

    Raises:
        FileNotFoundError: If the source or mask TIFF file is not found.
        rasterio.errors.RasterioIOError: If there is an error opening or writing the TIFF files.
    """
    # open the source tiff to get its transform and resolution
    with rasterio.open(source_tiff) as src:
        source_transform = src.transform
        source_crs = src.crs
        source_shape = src.shape

    # Reproject the mask in-memory to match the source TIFF
    with rasterio.open(mask_tiff) as mask_src:
        mask_data = mask_src.read(1)
        reprojected_mask = np.empty(source_shape, dtype=mask_data.dtype)

        reproject(
            source=mask_data,
            destination=reprojected_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=source_transform,
            dst_crs=source_crs,
            resampling=Resampling.nearest,
        )

        # Convert reprojected mask to boolean type as it is a mask
        reprojected_mask = reprojected_mask.astype(bool)

    return reprojected_mask


def get_log_file_name(base_name="conversion.log"):
    """Generate a new log file name with an incremented number if the base name already exists."""
    if not os.path.exists(base_name):
        return base_name
    base, ext = os.path.splitext(base_name)
    counter = 1
    new_name = f"{base}_{counter}{ext}"
    while os.path.exists(new_name):
        counter += 1
        new_name = f"{base}_{counter}{ext}"
    return new_name


def read_roi_from_config(filepath, roi_id: str = ""):
    gdf = gpd.read_file(filepath)
    if "type" in gdf.columns:
        gdf = gdf[gdf["type"] == "roi"].copy()
        if roi_id:
            gdf = gdf[gdf["id"] == roi_id].copy()
        else:
            return gdf
    return gdf


def remove_coregister_files(
    directory: str, suffix: str = "3B_TOAR_processed_coregistered.tif"
):
    inputs_paths = glob.glob(os.path.join(directory, f"*{suffix}"))
    for input_path in inputs_paths:
        os.remove(input_path)
        print(f"Removed {input_path}")


def convert_directory_to_model_format(
    directory: str,
    input_suffix: str = "*_TOAR.tif",
    output_suffix: str = "_TOAR_model_format.tif",
    separator="_3B",
    crs: int = None,
    verbose: bool = False,
    number_of_bands: int = 3,
    save_path: str = "",
):
    """
    Convert all files with a specific suffix in a directory to a specific TOAR model format.

    Args:
        directory (str): The directory path where the files are located.
        input_suffix (str, optional): The suffix of the input files to be converted. Defaults to '*_TOAR.tif'.
        output_suffix (str, optional): The suffix of the output files in the TOAR model format. Defaults to '_TOAR_model_format.tif'.
        crs (int, optional): The coordinate reference system to use for the output files. Defaults to None.
        separator (str, optional): The separator used in the filenames. Defaults to '_3B'.
        save_path (str, optional): The directory path to save the output files. Defaults to ''.
        verbose (bool, optional): If True, print the conversion process. Defaults to False.
        number_of_bands (int, optional): The number of bands in the output file. Defaults to 3.

    Returns:
        None
    """
    if save_path == "":
        save_path = directory
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    inputs_paths = glob.glob(os.path.join(directory, f"*{input_suffix}"))
    if len(inputs_paths) == 0:
        print(f"No files found with suffix {input_suffix} in {directory}")
        return

    for target_path in tqdm.tqdm(inputs_paths, desc="Converting files to model format"):
        base_filename = get_base_filename(target_path, separator)
        output_path = os.path.join(
            save_path, f"{base_filename}{separator}{output_suffix}"
        )

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            convert_planet_to_model_format(
                target_path, output_path, number_of_bands=number_of_bands, crs=crs
            )
            if verbose:
                print(f"Converting to model format and saving to {output_path}")
        except Exception as e:
            print(f"Error converting {target_path}: {e}")
            continue


def convert_file_to_toar(
    target_path, planet_dir, output_suffix: str = "_TOAR.tif", separator="_3B"
):
    """
    Converts the given target file to TOAR format using its associated XML metadata file.

    Parameters:
    - target_path: Path to the target file to be converted.
    - planet_dir: Directory where the planet files are stored.
    - separator: Separator used in the base filename.
    """

    # Function to get the base filename from the target path
    def get_base_filename(path, sep):
        return os.path.basename(path).split(sep)[0]

    base_filename = get_base_filename(target_path, separator)
    output_path = os.path.join(planet_dir, f"{base_filename}{separator}{output_suffix}")
    xml_files = glob.glob(
        os.path.join(planet_dir, f"{base_filename}*AnalyticMS_metadata_clip.xml")
    )

    if not xml_files:
        print(f"Could not find XML file for {target_path}")
        return

    xml_path = xml_files[0]
    print(f"Converting {target_path} to TOAR format and saving to {output_path}")
    # Assuming TOA_conversion is a predefined function
    TOA_conversion(target_path, xml_path, output_path)


def convert_directory_to_TOAR(
    directory: str,
    input_suffix: str = "*AnalyticMS_clip.tif",
    output_suffix: str = "_TOAR.tif",
    separator="_3B",
):
    """
    Converts all files in a directory to TOAR format.

    Args:
        directory (str): The directory path where the files are located.
        input_suffix (str, optional): The suffix of the input files. Defaults to '*AnalyticMS_clip.tif'.
        output_suffix (str, optional): The suffix of the output files. Defaults to '_TOAR.tif'.
        separator (str, optional): The separator used in the file names. Defaults to '_3B'.
    """
    inputs_paths = glob.glob(os.path.join(directory, f"*{input_suffix}"))
    if len(inputs_paths) == 0:
        print(f"No files found with suffix {input_suffix} in {directory}")
        return
    if len(inputs_paths) == 1:
        convert_file_to_toar(inputs_paths[0], directory, output_suffix, separator)
    for target_path in inputs_paths:
        convert_file_to_toar(target_path, directory, output_suffix, separator)


def create_elevation_mask(tiff_file, output_path, low_threshold, high_threshold):
    """
    Create an elevation mask based on a TIFF file.

    Args:
        tiff_file (str): The path to the input TIFF file.
        output_path (str): The path to save the output mask TIFF file.
        low_threshold (float): The lower threshold value for the mask.
        high_threshold (float): The upper threshold value for the mask.

    Returns:
        str: The path to the output mask TIFF file.
    """
    with rasterio.open(tiff_file) as src:
        mask_data = src.read(1)  # Read the first band

    # mask_data = np.flipud(mask_data)  # Flip the mask data vertically
    mask = np.logical_and(mask_data > low_threshold, mask_data < high_threshold)
    # save the mask to a new tiff file
    kwargs = src.meta
    kwargs.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, "w", **kwargs) as dst:
        dst.write_band(1, mask.astype(rasterio.uint8))
    return output_path


def read_topobathy_mask(mask_path: str) -> np.ndarray:
    mask_data = rasterio.open(mask_path).read(1)
    mask_data = mask_data.astype(bool)  # this is because it is a mask
    return mask_data


def match_resolution_and_grid(
    source_tiff: str, target_tiff: str, output_tiff: str
) -> str:
    """
    Matches the resolution and grid of a source TIFF file to a target TIFF file.

    Parameters:
        source_tiff (str): The path to the source TIFF file.
        target_tiff (str): The path to the target TIFF file.
        output_tiff (str): The path to the output TIFF file.

    Returns:
        str: The path to the output TIFF file.

    Raises:
        FileNotFoundError: If the source or target TIFF file is not found.
        rasterio.errors.RasterioIOError: If there is an error opening or writing the TIFF files.
    """
    # open the target tiff to get its transform and resolution
    with rasterio.open(target_tiff) as target:
        target_transform = target.transform
        target_crs = target.crs
        target_shape = target.shape

    # open the source tiff and update its metadata to match the target
    with rasterio.open(source_tiff) as src:
        src_meta = src.meta.copy()
        src_meta.update(
            {
                "crs": target_crs,
                "transform": target_transform,
                "width": target_shape[1],
                "height": target_shape[0],
            }
        )

        # create the output tiff and reproject the data
        with rasterio.open(output_tiff, "w", **src_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )

    return output_tiff


def get_pixel_resolution_and_crs(raster_path):
    with rasterio.open(raster_path) as dataset:
        # resolution is a tuple (pixel_width, pixel_height)
        resolution = dataset.res
        crs = dataset.crs
        return resolution, crs


def reproject_to_utm(input_tiff, output_tiff):
    # open the source tiff
    with rasterio.open(input_tiff) as src:
        # get the source crs
        src_crs = src.crs

        # get the bounds of the raster
        bounds = src.bounds
        lon = (bounds.left + bounds.right) / 2
        lat = (bounds.top + bounds.bottom) / 2

        # determine the utm zone
        utm_zone = (int((lon + 180) / 6) % 60) + 1
        is_northern = lat >= 0
        utm_crs = pyproj.CRS.from_dict(
            {
                "proj": "utm",
                "zone": utm_zone,
                "datum": "WGS84",
                "south": not is_northern,
            }
        )

        # calculate the transform and new dimensions
        transform, width, height = calculate_default_transform(
            src_crs, utm_crs, src.width, src.height, *src.bounds
        )

        # update the metadata
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": utm_crs, "transform": transform, "width": width, "height": height}
        )

        # create the output tiff and reproject the data
        with rasterio.open(output_tiff, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest,
                )
        return output_tiff


def format_landsat_tiff(landsat_path: str, output_dir: str = "") -> str:
    """
    Formats a Landsat TIFF file by converting it from float to uint16, changing the blocksize to 256x256,
    creating a tiled raster, and converting it to a 3 band RGB ordered TIFF.

    Args:
        landsat_path (str): The file path of the Landsat TIFF file.
        output_dir (str): The directory to save the formatted Landsat TIFF file.

    Returns:
        str: The file path of the formatted Landsat TIFF file.
    """
    # Output file path
    tmp_path = landsat_path.replace("_ms.tif", "_temp.tif")
    landsat_processed_path = landsat_path.replace("_ms.tif", "_processed.tif")
    # convert the landsat from float to uint16 and change the blocksize to 256x256
    tmp_path = convert_from_float_to_unit16(landsat_path, tmp_path)
    # create a tiled raster of the landsat image
    landsat_processed_path = create_tiled_raster(tmp_path, landsat_processed_path)
    output_path = landsat_processed_path.replace(".tif", "_model_format.tif")
    # create the output path at the output directory if specified
    if output_dir:
        output_path = os.path.join(output_dir, os.path.basename(output_path))
    # convert the landsat to a 3 band RGB ordered tiff
    convert_landsat_to_model_format(landsat_processed_path, output_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if os.path.exists(landsat_processed_path):
        os.remove(landsat_processed_path)
    return output_path


def read_raster(raster):
    with rasterio.open(raster) as src:
        return src.read(1)


def get_cloud_cover(filepath):
    """
    Calculate the cloud cover percentage for a given raster file.

    Parameters:
    filepath (str): The path to the raster file.

    Returns:
    tuple: A tuple containing the filepath and the cloud cover percentage.

    """
    mask = read_raster(filepath)
    mask_area = mask.shape[0] * mask.shape[1]
    cloud_mask = get_cloud_mask_landsat(mask)
    mask_sum = np.sum(cloud_mask)
    cloud_cover = np.round(mask_sum / mask_area, 3)
    return filepath, cloud_cover


def get_cloud_mask_landsat(im_QA: np.ndarray) -> np.ndarray[bool]:
    """
    Generate a cloud mask for Landsat imagery based on the QA band.

    This function specifically works for collection 2 Landsat data.

    Masks out cloud pixels indicated by the QA band values:
        dilated cloud = bit 1
        cirrus = bit 2
        cloud = bit 3

    This code was adapted from the code provided by the kvos in coastsat.
    Args:
        im_QA (np.ndarray): The QA band of the Landsat imagery.

    Returns:
        np.ndarray[bool]: A boolean array representing the cloud mask, where True indicates cloud pixels.

    Notes:
        - This function works for collection 2 Landsat data.
        - The cloud mask is generated based on the QA band values.
        - The function removes cloud pixels that form very thin features, which are often beach or swash pixels
          erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    """

    # function to return flag for n-th bit
    def is_set(x, n):
        return x & 1 << n != 0

    # dilated cloud = bit 1
    # cirrus = bit 2
    # cloud = bit 3
    qa_values = np.unique(im_QA.flatten())
    cloud_values = []
    for qaval in qa_values:
        for k in [1, 2, 3]:  # check the first 3 flags
            if is_set(qaval, k):
                cloud_values.append(qaval)

    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These are beach or swash pixels that are
    # erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        cloud_mask = morphology.remove_small_objects(
            cloud_mask, min_size=40, connectivity=1
        )
    return cloud_mask


def read_acc_georef(filepath):
    """
    Read the acc_georef value from a file.

    Parameters:
    filepath (str): The path to the file.

    Returns:
    tuple: A tuple containing the filepath and the acc_georef value.

    """
    acc_georef = 12  # default value
    with open(filepath, "r") as file:
        for line in file:
            if line.startswith("acc_georef"):
                acc_georef = float(line.split()[1])
                break

    return filepath, acc_georef


def get_date_acquired(filepath):
    filename_txt = os.path.basename(filepath)
    return filename_txt.split("_")[0]


def get_files_with_cloud_cover(directory):
    """
    Finds the file with the lowest accuracy georeference in the given directory.

    Args:
        directory (str): The directory path where the files are located.

    Returns:
        str: The file path of the file with the lowest accuracy georeference.
    """
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".tif")
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_cloud_cover, files)
    # return RMSE_files

    cloud_covered_files = []
    # create a dictionary with filename as key and acc_georef as value
    for file, cloud_cover in results:
        d = {"date_acquired": get_date_acquired(file), "cloud_cover": cloud_cover}
        # add the dictionary to the acc_georef_dict
        cloud_covered_files.append(d)
    return cloud_covered_files


def find_file_with_lowest_acc_georef(directory):
    """
    Finds the file with the lowest accuracy georeference in the given directory.

    Args:
        directory (str): The directory path where the files are located.

    Returns:
        str: The file path of the file with the lowest accuracy georeference.
    """
    files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".txt")
    ]
    print(len(files))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(read_acc_georef, files)

    RMSE_files = []
    # create a dictionary with filename as key and acc_georef as value
    for file, acc_georef in results:
        d = {"date_acquired": get_date_acquired(file), "acc_georef": acc_georef}
        # add the dictionary to the rmse_dict
        RMSE_files.append(d)
    return RMSE_files


def get_best_file(
    files, acc_georef_weight=0.7, cloud_cover_weight=0.3, cloud_cover_threshold=0.05
):
    """
    Returns the file with the lowest weighted combined acc_georef and cloud cover percentage.
    Ranks files higher if their cloud cover is below a specified threshold.

    Parameters:
        files (list of dict): A list where each dict contains 'filename', 'acc_georef', and 'cloud_cover'.
        acc_georef_weight (float): Weight for the acc_georef score (default is 0.5).
        cloud_cover_weight (float): Weight for the cloud cover score (default is 0.5).
        cloud_cover_threshold (float): The cloud cover threshold to rank files higher (default is 20).

    Returns:
        dict: The file with the best (lowest weighted combined) acc_georef and cloud cover percentage.
    """
    if not files:
        return None

    # Normalize acc_georef and cloud cover to the same scale
    max_acc_georef = max(file["acc_georef"] for file in files)
    max_cloud_cover = max(file["cloud_cover"] for file in files)

    def score(file):
        norm_acc_georef = file["acc_georef"] / max_acc_georef
        norm_cloud_cover = file["cloud_cover"] / max_cloud_cover
        combined_score = (norm_acc_georef * acc_georef_weight) + (
            norm_cloud_cover * cloud_cover_weight
        )

        # Adjust score to rank higher if cloud cover is below the threshold
        if file["cloud_cover"] < cloud_cover_threshold:
            combined_score *= (
                0.8  # Adjust this factor as needed to prioritize low cloud cover
            )

        return combined_score

    best_file = min(files, key=score)
    return best_file


def get_best_landsat_from_dir(landsat_dir: str) -> str:
    """
    Get the best Landsat from the given directory.

    Args:
        landsat_dir (str): The directory to search for Landsat files.

    Returns:
        str: The best Landsat found in the directory.

    Raises:
        ValueError: If a valid Landsat is not found in the directory.
    """
    # Define the ranking of Landsat versions in terms of quality
    landsat_ranking = {1: "L9", 2: "L8", 3: "L7", 4: "L5"}

    # Search for the best Landsat in the directory
    for landsat in landsat_ranking.values():
        if landsat in os.listdir(landsat_dir):
            # validate all the subdirectories are present
            if all(
                [
                    os.path.exists(os.path.join(landsat_dir, landsat, subdir))
                    for subdir in ["meta", "ms", "mask"]
                ]
            ):
                return landsat

    # Raise an error if a valid Landsat is not found
    raise ValueError(f"Could not find a valid Landsat in {landsat_dir}")


def get_best_landsat_tifs(session_directory: str) -> str:
    """
    Retrieves the paths of the best Landsat TIFF files based on the lowest accuracy georeference and lowest cloud cover.

    Args:
        session_directory (str): The directory path where the Landsat files are located.

    Returns:
        Tuple[str, str]: A tuple containing the paths of the best Landsat file and its corresponding cloud mask file.
    """
    landsat_dir = get_best_landsat_from_dir(session_directory)
    # # the available directories in the landsat directory
    meta_dir = os.path.join(session_directory, landsat_dir, "meta")
    ms_dir = os.path.join(session_directory, landsat_dir, "ms")
    mask_dir = os.path.join(session_directory, landsat_dir, "mask")

    cloud_list = get_files_with_cloud_cover(mask_dir)
    meta_list = find_file_with_lowest_acc_georef(meta_dir)

    # join the matching dictionaries in both list
    for cloud in cloud_list:
        for meta in meta_list:
            if cloud["date_acquired"] == meta["date_acquired"]:
                cloud.update(meta)

    # this would be the file with the lowest acc_georef and the lowest cloud cover
    best_file_stats = get_best_file(
        cloud_list
    )  # retuns a dictionary with the best file date_acquired, acc_georef and cloud_cover
    best_file_regex = f'{best_file_stats["date_acquired"]}*.tif'
    # get the best file
    best_landsat_path = glob.glob(os.path.join(ms_dir, f"{best_file_regex}"))[0]
    best_landsat_cloud_mask_path = glob.glob(
        os.path.join(mask_dir, f"{best_file_regex}")
    )[0]
    return best_landsat_path, best_landsat_cloud_mask_path


def read_tiff(input_path: str) -> np.ndarray:
    """
    Reads a TIFF image file and returns it as a NumPy array.

    Parameters:
        input_path (str): The path to the TIFF image file.

    Returns:
        np.ndarray: The image data as a NumPy array with shape (height, width, bands).
    """
    with rasterio.open(input_path) as src:
        # reads the image in band, x, y format
        img = src.read()
        # change the format to x, y, band
        img = np.moveaxis(img, 0, -1)
        return img


def convert_tiff_to_jpg(tiff_array: np.ndarray, filepath: str) -> None:
    """
    Convert a TIFF image array to a JPEG image and save it.

    Args:
        tiff_array (np.ndarray): The input TIFF image array.
        filepath (str): The path of the input TIFF file.

    Returns:
        None
    """
    if tiff_array.shape[2] < 3:
        raise Exception("Image does not have 3 bands, cannot be saved as a jpg")
    # Make a numpy array that can be saved as a jpg
    im_RGB = tiff_array[:, :, 0:3]
    im_RGB = img_as_ubyte(im_RGB)
    fname = filepath.replace(".tif", ".jpg")
    print(f"Saving image to {fname}")
    imsave(fname, im_RGB, quality=100)


def convert_landsat_to_model_format(input_file: str, output_file: str) -> None:
    """Process the raster file by normalizing and reordering its bands, and save the output.
    It removes the 5th band (SWIR) and reorders the bands to RGBN (red, green, blue, NIR).

    This is used for 4 band planet imagery that needs to be reordered to RGBN from BGRN for the zoo model.

    Args:
        input_file (str): The file path to the input raster file.
        output_file (str): The file path to save the processed raster file.

    Reads the input raster file, normalizes its bands to the range [0, 255],
    reorders the bands to RGB followed by the remaining bands, and saves the
    processed raster to the specified output file.

    The function assumes the input raster has at least three bands (blue, green, red)
    and possibly additional bands.

    Prints the min and max values of the original and normalized red, green, and blue bands,
    and the shape of the reordered bands array.
    """
    # remove the output file if exists just in case it was corrupted (opening with rasterio will fail)
    if os.path.exists(output_file):
        os.remove(output_file)

    with rasterio.open(input_file) as src:
        # Read the bands
        band1 = src.read(1)  # blue
        band2 = src.read(2)  # green
        band3 = src.read(3)  # red
        other_bands = [src.read(i) for i in range(4, src.count + 1)]

        # Normalize the bands
        band1_normalized = normalize_band(band1)
        band2_normalized = normalize_band(band2)
        band3_normalized = normalize_band(band3)
        other_bands_normalized = [normalize_band(band) for band in other_bands]

        # Reorder the bands RGB and other bands
        # reordered_bands = np.dstack([band1_normalized, band2_normalized,band3_normalized,] + other_bands_normalized)
        reordered_bands = np.dstack(
            [
                band3_normalized,
                band2_normalized,
                band1_normalized,
            ]
            + other_bands_normalized
        )
        reordered_bands = reordered_bands[:, :, :3]
        # Get the metadata
        meta = src.meta.copy()

        # print(
        #     f"dtype: {reordered_bands.dtype}, shape: {reordered_bands.shape}, count: {reordered_bands.shape[2]}"
        # )
        # Update the metadata to reflect the number of layers and data type
        meta.update(
            {
                "count": reordered_bands.shape[2],
                "dtype": reordered_bands.dtype,
                "driver": "GTiff",
            }
        )
        # make a numpy array that can be saved as a jpg
        # convert_tiff_to_jpg(reordered_bands, input_file)

        # Save the image
        with rasterio.open(output_file, "w", **meta) as dst:
            for i in range(reordered_bands.shape[2]):
                dst.write(reordered_bands[:, :, i], i + 1)

    return output_file


def create_geometry(
    geomtype: str, shoreline: List[List[float]]
) -> Optional[Union[geometry.LineString, geometry.MultiPoint]]:
    """
    Creates geometry based on geomtype and shoreline data.

    Parameters:
    -----------
    geomtype: str
        Type of geometry ('lines' or 'points').
    shoreline: List[List[float]]
        List of shoreline coordinates.
        Ex: [[0,1],[1,0]]

    Returns:
    --------
    Union[geometry.LineString, geometry.MultiPoint, None]
        The created geometry or None if invalid.
    """
    if geomtype == "lines" and len(shoreline) >= 2:
        return geometry.LineString(shoreline)
    elif geomtype == "points" and len(shoreline) > 0:
        return geometry.MultiPoint([(coord[0], coord[1]) for coord in shoreline])
    return None


def create_gdf_from_shoreline(
    shoreline: List[List[float]], output_epsg: int, date: str, geomtype: str = "lines"
):
    """
    Create a GeoDataFrame from a shoreline.

    Args:
        shoreline (List[List[float]]): A list of coordinates representing the shoreline.
        output_epsg (int): The EPSG code of the output coordinate reference system.
        date (str): The date associated with the shoreline.
        geomtype (str, optional): The type of geometry to create. Defaults to "lines".

    Returns:
        GeoDataFrame: A GeoDataFrame containing the shoreline geometry and associated attributes.
        None: If the shoreline geometry cannot be created.

    """
    geom = create_geometry(geomtype, shoreline)
    if geom:
        # Convert date string to datetime object
        date = datetime.strptime(date, "%Y-%m-%d-%H-%M-%S")
        # Creating a GeoDataFrame directly with all attributes
        shoreline_gdf = gpd.GeoDataFrame(
            {"date": [date]}, geometry=[geom], crs=f"EPSG:{output_epsg}"
        )
        return shoreline_gdf
    return None


def get_date_from_path(filename):
    # Original string
    "_".join(filename.split("_")[:2])

    # Convert to datetime object
    dt = datetime.strptime("_".join(filename.split("_")[:2]), "%Y%m%d_%H%M%S")

    return dt.strftime("%Y-%m-%d-%H-%M-%S")


def get_epsg_from_tiff(tiff_path):
    """
    Retrieves the EPSG code from a TIFF file.

    Args:
        tiff_path (str): The path to the TIFF file.

    Returns:
        None

    Prints the EPSG code if found, or "No EPSG found" if not.
    """
    data = gdal.Open(tiff_path, gdal.GA_ReadOnly)
    projection = data.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    if srs.IsProjected:
        print(srs.GetAuthorityCode(None))
        return srs.GetAuthorityCode(None)
    else:
        print("No EPSG found")
        return None


# function from coastsat_plantscope
def TOA_conversion(image_path, xml_path, save_path):
    """
    1) Convert DN values to Top of Atmosphere (TOA)
    2) Add sensor type (PS2, PS2.SD, PSB.SD) to save filename

    Function modified from:
        https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/toar/toar_planetscope.ipynb

    """

    # Load image bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(image_path) as src:
        band_blue_radiance = src.read(1)

    with rasterio.open(image_path) as src:
        band_green_radiance = src.read(2)

    with rasterio.open(image_path) as src:
        band_red_radiance = src.read(3)

    with rasterio.open(image_path) as src:
        band_nir_radiance = src.read(4)

    ### Get TOA Factor ###
    xmldoc = minidom.parse(xml_path)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ["1", "2", "3", "4"]:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[
                0
            ].firstChild.data
            coeffs[i] = float(value)

    # print("Conversion coefficients: {}".format(coeffs))

    ### Convert to TOA ###

    # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
    band_blue_reflectance = band_blue_radiance * coeffs[1]
    band_green_reflectance = band_green_radiance * coeffs[2]
    band_red_reflectance = band_red_radiance * coeffs[3]
    band_nir_reflectance = band_nir_radiance * coeffs[4]

    # print("Red band radiance is from {} to {}".format(np.amin(band_red_radiance), np.amax(band_red_radiance)))
    # print("Red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))

    # find sensor name
    node = xmldoc.getElementsByTagName("eop:Instrument")
    sensor = node[0].getElementsByTagName("eop:shortName")[0].firstChild.data

    print("Sensor: {}".format(sensor))

    ### Save output images ###

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(dtype=rasterio.uint16, count=4)

    # print("Before Scaling, red band reflectance is from {} to {}".format(np.amin(band_red_reflectance), np.amax(band_red_reflectance)))
    # Here we include a fixed scaling factor. This is common practice.
    scale = 10000
    blue_ref_scaled = scale * band_blue_reflectance
    green_ref_scaled = scale * band_green_reflectance
    red_ref_scaled = scale * band_red_reflectance
    nir_ref_scaled = scale * band_nir_reflectance

    # print("After Scaling, red band reflectance is from {} to {}".format(np.amin(red_ref_scaled), np.amax(red_ref_scaled)))
    print(f"Saving to: {os.path.abspath(save_path)}")
    # Write band calculations to a new raster file
    with rasterio.open(save_path, "w", **kwargs) as dst:
        dst.write_band(1, blue_ref_scaled.astype(rasterio.uint16))
        dst.write_band(2, green_ref_scaled.astype(rasterio.uint16))
        dst.write_band(3, red_ref_scaled.astype(rasterio.uint16))
        dst.write_band(4, nir_ref_scaled.astype(rasterio.uint16))


def get_base_filename(tiff_path, separator="_3B"):
    if separator not in os.path.basename(tiff_path):
        raise ValueError(f"Separator '{separator}' not found in '{tiff_path}'")
    return os.path.basename(tiff_path).split(separator, 1)[0]


def calculate_area_from_bounds(bounds):
    """
    Calculate the area from the bounding box.

    Args:
    - bounds (BoundingBox): The bounding box.

    Returns:
    - float: The area of the bounding box in square meters.
    """
    return (bounds.right - bounds.left) * (bounds.top - bounds.bottom)


def square_meters_to_square_kilometers(area_sq_meters):
    """
    Convert area from square meters to square kilometers.

    Args:
    - area_sq_meters (float): Area in square meters.

    Returns:
    - float: Area in square kilometers.
    """
    return area_sq_meters / 1_000_000


def square_kilometers_to_square_meters(area_sq_kilometers):
    """
    Convert area from square kilometers to square meters.

    Args:
    - area_sq_kilometers (float): Area in square kilometers.

    Returns:
    - float: Area in square meters.
    """
    return area_sq_kilometers * 1_000_000


def check_tiff_area(tiff_path, expected_area_km, threshold=0.8):
    """
    Check if the TIFF file area (using bounds) is above a certain threshold of the expected area.

    Args:
    - tiff_path (str): Path to the TIFF file.
    - expected_area_km (float): Expected area in square kilometers.
    - threshold (float): The threshold percentage (default is 0.8 for 80%).

    Returns:
    - tuple: (tiff_path, bool, float) Path to the TIFF file, whether the area is above the threshold, and the area of the TIFF file in square kilometers.
    """
    try:
        with rasterio.open(tiff_path) as src:
            tiff_bounds = src.bounds
    except rasterio.errors.RasterioIOError:
        print(f"Error reading {os.path.basename(tiff_path)}")
        return tiff_path, False, 0
    tiff_area_m = calculate_area_from_bounds(tiff_bounds)
    tiff_area_km = square_meters_to_square_kilometers(tiff_area_m)
    expected_area_m = square_kilometers_to_square_meters(expected_area_km)

    is_above_threshold = tiff_area_m >= (expected_area_m * threshold)

    return tiff_path, is_above_threshold, tiff_area_km


def get_tiffs_with_bad_area(
    tiff_paths, expected_area_km, threshold=0.8, use_threads=True
):
    if use_threads:
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        executor_class = concurrent.futures.ProcessPoolExecutor

    filtered_tiffs = []

    with executor_class() as executor:
        future_to_tiff = {
            executor.submit(
                check_tiff_area, tiff_path, expected_area_km, threshold
            ): tiff_path
            for tiff_path in tiff_paths
        }

        for future in concurrent.futures.as_completed(future_to_tiff):
            tiff_path, is_above_threshold, tiff_area_km = future.result()
            if not is_above_threshold:
                filtered_tiffs.append(tiff_path)

    return filtered_tiffs


def filter_tiffs_by_area(tiff_paths, expected_area_km, threshold=0.8, use_threads=True):
    """
    Filters a list of TIFF file paths based on their area.

    Args:
        tiff_paths (list): A list of file paths to TIFF files.
        expected_area_km (float): The expected area in square kilometers.
        threshold (float, optional): The threshold value for determining if a TIFF file's area is above the expected area. Defaults to 0.8.
        use_threads (bool, optional): Specifies whether to use threads for parallel execution. Defaults to True.

    Returns:
        list: A list of filtered TIFF file paths that have an area above the expected area.
    """
    if use_threads:
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        executor_class = concurrent.futures.ProcessPoolExecutor

    filtered_tiffs = []

    with executor_class() as executor:
        future_to_tiff = {
            executor.submit(
                check_tiff_area, tiff_path, expected_area_km, threshold
            ): tiff_path
            for tiff_path in tiff_paths
        }

        for future in concurrent.futures.as_completed(future_to_tiff):
            tiff_path, is_above_threshold, tiff_area_km = future.result()
            if is_above_threshold:
                filtered_tiffs.append(tiff_path)

    return filtered_tiffs


def filter_tiffs_by_area_no_threads(tiff_paths, expected_area_km, threshold=0.8):
    filtered_tiffs = []

    for tiff_path in tiff_paths:
        tiff_path, is_above_threshold, tiff_area_km = check_tiff_area(
            tiff_path, expected_area_km, threshold
        )

        if is_above_threshold:
            filtered_tiffs.append(tiff_path)

    return filtered_tiffs


def read_planet_tiff(planet_path: str, num_bands: list) -> np.ndarray:
    """
    Read a Planet TIFF image and return it as a numpy array.

    Args:
        planet_path (str): The file path to the Planet TIFF image.
        num_bands (list): A list of band indices to read from the image.

    Returns:
        numpy.ndarray: The image data as a numpy array with shape (rows, cols, bands).
    """
    # read in the planet image as an RGB image
    with rasterio.open(planet_path) as src:
        im_ms = src.read(num_bands)
        # reorder the numpy array from (bands, rows, cols) to (rows, cols, bands)
        im_ms = np.moveaxis(im_ms, 0, -1)
        return im_ms


def get_georef(fn):
    """
    Retrieves the georeferencing information from a given file.

    Parameters:
    fn (str): The file path of the input file.

    Returns:
    numpy.ndarray: An array containing the georeferencing information.

    """
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    return georef


def create_tiled_raster(src_filepath, dst_filepath, block_size=256):
    """
    Create a new tiled raster file with the specified block size.

    Parameters:
        src_filepath (str): The file path of the source raster file.
        dst_filepath (str): The file path of the destination raster file.
        block_size (int, optional): The size of the blocks for tiling. Defaults to 256.

    Returns:
        dst_filepath (str): The file path of the destination raster file.

    Raises:
        None
    """

    # Open the source file
    with rasterio.open(src_filepath) as src:
        # Copy the profile from the source
        profile = src.profile

        # Update the profile to set tiling and block sizes
        profile.update(
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            compress="lzw",  # Optional: Add compression to reduce file size
        )

        # remove the destination file if it already exists just in case it was corrupted
        if os.path.exists(dst_filepath):
            os.remove(dst_filepath)

        # Create a new file with the updated profile
        with rasterio.open(dst_filepath, "w", **profile) as dst:
            # Copy data from the source file to the destination file
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                dst.write(band_data, i)
    return dst_filepath


def convert_from_float_to_unit16(input_path, output_path):
    """
    Process a raster file by performing the following steps:
    1. Open the input raster file.
    2. Modify the data by replacing infinite values with 0 and scaling the data by a factor of 10000.
    3. Create a new output raster file with the modified data.

    Args:
        input_path (str): The path to the input raster file.
        output_path (str): The path to save the output raster file.

    Returns:
        str: The path to the output raster file.
    """
    if os.path.isfile(output_path) and os.path.exists(output_path):
        os.remove(output_path)

    with rasterio.open(input_path) as input_raster:
        out_meta = input_raster.meta.copy()
        modified_data = np.empty(
            (input_raster.count, input_raster.height, input_raster.width),
            dtype=out_meta["dtype"],
        )
        scale = 10000
        for i in range(1, input_raster.count + 1):
            data = input_raster.read(i)
            data[np.isinf(data)] = 0
            data = data * scale
            modified_data[i - 1] = data

        out_meta.update(dtype=rasterio.uint16, count=input_raster.count, nodata=0)

    with rasterio.open(output_path, "w", **out_meta) as output_raster:
        output_raster.write(modified_data)
    return output_path


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize the band to the range [0, 255].

    Args:
        band (np.ndarray): The input band to normalize.

    Returns:
        np.ndarray: The normalized band.
    """
    min_val = np.min(band)
    max_val = np.max(band)
    normalized_band = ((band - min_val) / (max_val - min_val)) * 255
    return normalized_band.astype(np.uint8)


# def normalize_band(band):
#     """Normalize the band to the range [0, 255] and handle infinities."""
#     # Replace infinities with NaN for easier handling
#     band = np.where(np.isinf(band), np.nan, band)
#     min_val = np.nanmin(band)
#     max_val = np.nanmax(band)
#     # Replace NaNs with the min value for normalization purposes
#     band = np.where(np.isnan(band), min_val, band)
#     normalized_band = ((band - min_val) / (max_val - min_val)) * 255
#     return normalized_band.astype(np.uint8)


def reproject_raster(input_file: str, output_file: str, target_crs) -> None:
    """Reproject a raster to a new CRS if it is not already in that CRS and replace the original file.

    Args:
        input_file (str): The file path to the input raster file.
        target_crs (dict or str): The target coordinate reference system.
    """
    with rasterio.open(input_file) as src:
        if src.crs == target_crs:
            print(f"The raster is already in the target CRS: {target_crs}")
            return output_file
        else:
            print(f"Reprojecting the raster to {target_crs}")
            # Calculate the transform and dimensions for the new CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(output_file, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
    return output_file


def reproject_raster_in_place(input_file: str, target_crs) -> None:
    """Reproject a raster to a new CRS if it is not already in that CRS and replace the original file.

    Args:
        input_file (str): The file path to the input raster file.
        target_crs (dict or str): The target coordinate reference system.
    """
    temp_file = "temp_reprojected.tif"

    with rasterio.open(input_file) as src:
        if src.crs == target_crs:
            print(f"The raster is already in the target CRS: {target_crs}")
            return  # No need to reproject
        else:
            print(f"Reprojecting the raster to {target_crs}")
            # Calculate the transform and dimensions for the new CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            meta = src.meta.copy()
            meta.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            with rasterio.open(temp_file, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )

    # Replace the original file with the reprojected file
    os.remove(input_file)
    os.rename(temp_file, input_file)
    print(f"Reprojected raster saved as the original file: {input_file}")


def convert_planet_to_model_format(
    input_file: str, output_file: str, number_of_bands: int = 3, crs=None
) -> None:
    """Process the raster file by normalizing and reordering its bands, and save the output.

    This is used for 4 band planet imagery that needs to be reordered to RGBN from BGRN for the zoo model.

    Args:
        input_file (str): The file path to the input raster file.
        output_file (str): The file path to save the processed raster file.
        number_of_bands (int): The number of bands to keep in the output.
        crs (dict or str, optional): The target coordinate reference system.

    Reads the input raster file, normalizes its bands to the range [0, 255],
    reorders the bands to RGB followed by the remaining bands, and saves the
    processed raster to the specified output file.

    The function assumes the input raster has at least three bands (blue, green, red)
    and possibly additional bands.

    Prints the min and max values of the original and normalized red, green, and blue bands,
    and the shape of the reordered bands array.
    """
    temp_file = "reprojected_temp.tif"
    log_file = get_log_file_name("conversion.log")
    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    try:
        if crs:
            logging.info(f"Reprojecting the raster to {crs}")
            # just in case the temp file exists & is corrupted
            if os.path.exists(temp_file):
                os.remove(temp_file)

            with rasterio.open(input_file) as src:
                if src.crs == crs:
                    logging.info(f"The raster is already in the target CRS: {crs}")
                    reprojected_file = input_file
                else:
                    # Calculate the transform and dimensions for the new CRS
                    transform, width, height = calculate_default_transform(
                        src.crs, crs, src.width, src.height, *src.bounds
                    )
                    meta = src.meta.copy()
                    meta.update(
                        {
                            "crs": crs,
                            "transform": transform,
                            "width": width,
                            "height": height,
                        }
                    )

                    with rasterio.open(temp_file, "w", **meta) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=crs,
                                resampling=Resampling.nearest,
                            )

                    reprojected_file = temp_file
        else:
            reprojected_file = input_file

        # this prevents issues if the output file already exists and is corrupted
        print(f"output file: {output_file}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                logging.info(f"Existing output file {output_file} deleted.")
            except Exception as e:
                logging.error(
                    f"Could not delete existing output file {output_file}: {e}"
                )
                raise e

        with rasterio.open(reprojected_file) as src:
            # Read the bands
            band1 = src.read(1)  # blue
            band2 = src.read(2)  # green
            band3 = src.read(3)  # red
            other_bands = [src.read(i) for i in range(4, src.count + 1)]

            # Normalize the bands
            band1_normalized = normalize_band(band1)
            band2_normalized = normalize_band(band2)
            band3_normalized = normalize_band(band3)
            other_bands_normalized = [normalize_band(band) for band in other_bands]

            # Reorder the bands RGB and other bands
            reordered_bands = np.dstack(
                [band3_normalized, band2_normalized, band1_normalized]
                + other_bands_normalized
            )
            reordered_bands = reordered_bands[:, :, :number_of_bands]

            # Get the metadata
            meta = src.meta.copy()

            # Update the metadata to reflect the number of layers and data type
            meta.update(
                {
                    "count": reordered_bands.shape[2],
                    "dtype": reordered_bands.dtype,
                    "driver": "GTiff",
                }
            )

            # Save the image
            with rasterio.open(output_file, "w", **meta) as dst:
                for i in range(reordered_bands.shape[2]):
                    dst.write(reordered_bands[:, :, i], i + 1)

        # Delete the temporary file if it was created
        if crs and os.path.exists(temp_file):
            os.remove(temp_file)

        logging.info(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}", exc_info=True)
        raise e
