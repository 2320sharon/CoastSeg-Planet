import os
import re
import asyncio
from typing import List, Dict, Union, Set, Optional, Callable
import json
import xml.etree.ElementTree as ET
import rasterio
from rasterio.warp import transform_geom
from coastseg_planet.db.base import BaseDuckDB
from coastseg_planet.db.roi_repository import ROIRepository
from coastseg_planet.db.tile_repository import TileRepository
from coastseg_planet.db.order_repository import OrderRepository
from coastseg_planet.processor import TileProcessor
from coastseg_planet.config import DATABASE_PATH, TILE_STATUSES
import re


async def insert_order(folder_path: str, processor):
    """
    Attempts to insert an order into the database using the folder name and manifest.json file.

    Args:
        folder_path (str): Path to the order folder.
        processor: The TileProcessor instance used to interact with the database.
    """
    order_name = os.path.basename(folder_path)
    manifest_path = find_file(folder_path, "manifest.json")

    if not manifest_path:
        print(f"[INFO] No manifest.json found in {folder_path}. Skipping order insert.")
        return

    await processor.process(
        {
            "action": "update_order",
            "order_id": order_name,
            "status": "downloaded",
            "filepath": folder_path,
        }
    )
    print(f"[SUCCESS] Inserted order '{order_name}' with manifest.")


def search_files(directory, pattern, search_strings):
    """
    Recursively search through a directory for files matching a regex pattern.
    For each matching file, check if any of the provided strings are present in the file name.

    Args:
        directory (str): The root directory to start the search.
        pattern (str): The regex pattern to match file names (e.g. r'.*\\.(tiff?|xml)$').
        search_strings (Union[str, List[str]]): String or list of strings to search for in file names.

    Returns:
        bool: True if any matching file contains any of the search strings, otherwise False.
    """
    if isinstance(search_strings, str):
        search_strings = [search_strings]

    regex = re.compile(pattern, re.IGNORECASE)

    for root, _, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                # check if the the file name contains any of the search strings
                if any(s in file for s in search_strings):
                    print(f"Found matching file: {file}")
                    return True

    return False


def is_order_id(folder_name: str) -> bool:
    """Check if a folder name is a valid Planet order ID."""
    pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    return bool(pattern.match(folder_name))


def read_from_tif(file_file, geojson_path=None):
    """
    Reads the boundaries from the given GeoTIFF file, converts them to EPSG:4326,
    and returns a GeoJSON-style polygon.

    Parameters:
      file_file (str): Path to the input GeoTIFF file.
      geojson_path (str, optional): If provided, the GeoJSON output is saved to this file.

    Returns:
      dict: A GeoJSON Feature representing the polygon in EPSG:4326.
    """
    with rasterio.open(file_file) as src:
        # Print original CRS and bounds.
        print("Source CRS:", src.crs)
        bounds = src.bounds
        print("Original bounds:", bounds)

        # Create a polygon (in the source CRS) from the bounds.
        polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [bounds.left, bounds.bottom],
                    [bounds.right, bounds.bottom],
                    [bounds.right, bounds.top],
                    [bounds.left, bounds.top],
                    [bounds.left, bounds.bottom],
                ]
            ],
        }

        # Transform the polygon from the source CRS to EPSG:4326.
        transformed_polygon = transform_geom(src.crs, "EPSG:4326", polygon)

        geojson = {"type": "Feature", "geometry": transformed_polygon, "properties": {}}

        # Optionally, write the GeoJSON to a file.
        if geojson_path:
            with open(geojson_path, "w") as f:
                json.dump(geojson, f, indent=2)

        return geojson


def xml_to_geojson_polygon(xml_path, geojson_path=None):
    """
    Reads the footprint from the given XML file and converts it into a GeoJSON-style polygon.

    Parameters:
      xml_path (str): Path to the XML file.
      geojson_path (str, optional): If provided, the GeoJSON output will be saved to this file.

    Returns:
      dict: A GeoJSON Feature representing the footprint polygon.
    """
    # Define namespaces used in the XML file.
    ns = {
        "gml": "http://www.opengis.net/gml",
        "ps": "http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level",
    }

    # Parse the XML file.
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the coordinates string within the footprint element.
    coord_elem = root.find(
        ".//ps:Footprint/gml:multiExtentOf/gml:MultiSurface/gml:surfaceMembers/"
        "gml:Polygon/gml:outerBoundaryIs/gml:LinearRing/gml:coordinates",
        ns,
    )

    if coord_elem is None or not coord_elem.text:
        raise ValueError("Footprint coordinates not found in XML.")

    # The coordinates are provided as space-separated "lon,lat" pairs.
    coords_text = coord_elem.text.strip()
    coord_pairs = coords_text.split()
    polygon_coords = []

    for pair in coord_pairs:
        lon_str, lat_str = pair.split(",")
        polygon_coords.append([float(lon_str), float(lat_str)])

    # Ensure that the polygon is closed (first coordinate equals the last).
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    # Create a GeoJSON Feature.
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coords],  # GeoJSON expects a list of linear rings.
        },
        "properties": {},
    }

    # Optionally write the GeoJSON to a file.
    if geojson_path:
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


# Define extractor functions
extractors = {
    "json": lambda path: read_json(path).get("geometry", {}),
    "xml": lambda path: xml_to_geojson_polygon(path).get("geometry"),
    "analytic": lambda path: read_from_tif(path).get("geometry"),
    "udm": lambda path: read_from_tif(path).get("geometry"),
}


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


# def extract_geometry(metadata_file):
#     if metadata_file and os.path.exists(metadata_file):
#         metadata = read_json(metadata_file)
#         return metadata.get("geometry", {})
#     return {}


def extract_geometry(
    metadata_file: Optional[str],
    fallback_files: List[str],
    extractors: Dict[str, Callable[[str], Optional[dict]]],
) -> dict:
    """
    Attempts to extract GeoJSON geometry from a list of prioritized sources:
    1. Primary metadata JSON
    2. XML footprint files
    3. Analytic TIFF
    4. UDM TIFF

    Args:
        metadata_file (str or None): Path to the primary metadata file (usually a JSON).
        fallback_files (List[str]): List of all files for the tile.
        extractors (Dict[str, Callable[[str], Optional[dict]]]):
            A mapping of source type to extractor function.

    Returns:
        dict: A GeoJSON geometry dictionary or an empty dict if extraction fails.
    """

    def try_extract(
        file_path: str, extractor: Callable[[str], Optional[dict]]
    ) -> Optional[dict]:
        try:
            if os.path.exists(file_path):
                geometry = extractor(file_path)
                if geometry:
                    print(f"[INFO] Geometry extracted from: {file_path}")
                    return geometry
        except Exception as e:
            print(f"[WARN] Extraction failed for {file_path}: {e}")
        return None

    # 1. Attempt metadata JSON file first
    if metadata_file and os.path.exists(metadata_file):
        geometry = try_extract(metadata_file, extractors["json"])
        if geometry:
            return geometry

    # 2. Iterate through priority list and apply matching extractor
    priorities = [
        ("xml", lambda f: f.lower().endswith(".xml")),
        ("analytic", lambda f: "_analyticms.tif" in f.lower()),
        ("udm", lambda f: "_udm2.tif" in f.lower()),
    ]

    for source_type, match_fn in priorities:
        extractor = extractors.get(source_type)
        if not extractor:
            continue

        for file in fallback_files:
            if file == metadata_file:
                continue
            if match_fn(file):
                geometry = try_extract(file, extractor)
                if geometry:
                    return geometry

    print("[WARNING] No valid geometry found in any metadata or fallback files.")
    return {}


def extract_geometry_from_geojson(geojson_path):
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
    geojson = read_json(geojson_path)
    features = geojson.get("features", [])
    if not features:
        raise ValueError("GeoJSON file contains no features.")
    return features[0].get("geometry", {})


def extract_unique_datetime_ids(items: Union[str, List[str]]) -> Union[str, Set[str]]:
    """
    Extracts unique datetime IDs in the format 'YYYYMMDD_HHMMSS_XX' from a string or list of strings.

    Args:
        items (Union[str, List[str]]): A filename string or list of filename strings.

    Returns:
        Union[str, Set[str]]: A single datetime ID string if one match is found, otherwise a set of unique IDs.
    """
    datetime_pattern = re.compile(r"\d{8}_\d{6}_\d{2}")
    unique_ids = set()

    # Normalize to a list for uniform handling
    if isinstance(items, str):
        items = [items]

    for filename in items:
        match = datetime_pattern.search(filename)
        if match:
            unique_ids.add(match.group())

    if len(unique_ids) == 1:
        return next(iter(unique_ids))
    return unique_ids


# async def process_tile_mode(processor, id_dict):
#     # Insert each tile and its geometry
#     for tile_id, details in id_dict.items():
#         print(f"Processing tile {tile_id}...")
#         metadata_file = details["metadata"]
#         if not metadata_file:
#             print(f"Skipping tile {tile_id}: No metadata found.")
#             continue
#         geometry = extract_geometry(metadata_file)
#         print(f"Extracted geometry for tile {tile_id}: {geometry}")
#         # this inserts the tile into the database
#         await processor.process(
#             {
#                 "action": "insert_tile",
#                 "tile_id": tile_id,
#                 "capture_time": extract_unique_datetime_ids(tile_id),
#                 "geometry": geometry,
#             }
#         )

#     # Insert associated files after tiles (FK constraint)
#     for tile_id, details in id_dict.items():
#         for filepath in details["files"]:
#             await processor.process(
#                 {
#                     "action": "update_metadata_tile",
#                     "tile_id": tile_id,
#                     "order_id": None,
#                     "filepath": filepath,
#                     "status": TILE_STATUSES["DOWNLOADED"],
#                 }
#             )

#     print("✅ Inserted all tiles and files in tile mode.")


async def process_tile_mode(processor, id_dict, extractors, order_name=None):
    # Insert each tile and its geometry
    for tile_id, details in id_dict.items():
        print(f"Processing tile {tile_id}...")
        metadata_file = details.get("metadata")
        fallback_files = details.get("files", [])

        geometry = extract_geometry(metadata_file, fallback_files, extractors)
        if not geometry:
            print(f"Skipping tile {tile_id}: Geometry extraction failed.")
            continue

        print(f"Extracted geometry for tile {tile_id}: {geometry}")
        await processor.process(
            {
                "action": "insert_tile",
                "tile_id": tile_id,
                "capture_time": extract_unique_datetime_ids(tile_id),
                "geometry": geometry,
                "order_name": order_name,
            }
        )

    # Insert associated files after tiles (FK constraint)
    for tile_id, details in id_dict.items():
        print(f"Associating files with tile {tile_id}...")
        for filepath in details.get("files", []):
            print(f" tile_id: {tile_id}, filepath: {filepath}")
            await processor.process(
                {
                    "action": "update_metadata_tile",
                    "tile_id": tile_id,
                    "order_id": None,
                    "filepath": filepath,
                    "status": TILE_STATUSES["DOWNLOADED"],
                }
            )

    print("✅ Inserted all tiles and files in tile mode.")


# async def single_shared_ROI_method(shared_geometry_path, roi_name, id_dict, processor):
#     # # Determine geometry
#     # if shared_geometry_path:
#     #     shared_geometry = extract_geometry_from_geojson(shared_geometry_path)
#     # else:
#     #     # Use first available metadata
#     #     first_meta = next(
#     #         (
#     #             details["metadata"]
#     #             for details in id_dict.values()
#     #             if details["metadata"]
#     #         ),
#     #         None,
#     #     )
#     #     if not first_meta:
#     #         raise FileNotFoundError("No metadata file found to extract ROI geometry.")
#     #     shared_geometry = extract_geometry(first_meta)

#     # Insert one ROI into tiles table
#     await processor.process(
#         {
#             "action": "insert_roi",
#             "roi_id": roi_name,
#             "tile_id": "",
#             "capture_time": "",
#             "geometry": shared_geometry,
#         }
#     )

#     # Associate all files with the ROI ID
#     for tile_id, details in id_dict.items():
#         # for each tile make a new entry in the roi_tiles table
#         await processor.process(
#             {
#                 "action": "insert_roi_tile",
#                 "roi_id": roi_name,
#                 "tile_id": tile_id,
#                 "capture_time": extract_unique_datetime_ids(tile_id),
#                 "intersection": None,
#                 "fallback_geom": shared_geometry,
#             }
#         )
#         for filepath in details["files"]:
#             filename = os.path.basename(filepath)
#             await processor.process(
#                 {
#                     "action": "update_metadata_roi",
#                     "roi_id": roi_name,
#                     "tile_id": "_".join(filename.split("_")[:4]),
#                     "order_id": "",
#                     "filepath": filepath,
#                     "status": TILE_STATUSES["PENDING"],
#                 }
#             )


def extract_first_available_geometry(
    id_dict: Dict[str, Dict[str, Optional[str]]],
    extractors: Dict[str, Callable[[str], Optional[dict]]],
) -> dict:
    """
    Attempts to extract the first available geometry across all tiles.
    Priority: metadata JSON > XML > TIFF (Analytic > UDM)

    Args:
        id_dict: Dictionary of tile metadata and associated files.
        extractors: Dictionary of extractor functions for json/xml/tif types.

    Returns:
        dict: GeoJSON geometry or empty dict.
    """

    def try_all(
        files: List[str], match_fn: Callable[[str], bool], extractor_key: str
    ) -> Optional[dict]:
        extractor = extractors.get(extractor_key)
        if not extractor:
            return None
        for f in files:
            if match_fn(f):
                try:
                    if os.path.exists(f):
                        geometry = extractor(f)
                        if geometry:
                            print(
                                f"[INFO] Geometry extracted from {extractor_key.upper()} file: {f}"
                            )
                            return geometry
                except Exception as e:
                    print(f"[WARN] Failed to extract geometry from {f}: {e}")
        return None

    all_files = []
    for details in id_dict.values():
        if details.get("metadata"):
            all_files.append(details["metadata"])
        all_files.extend(details.get("files", []))

    # 1. Try all metadata JSONs first
    geometry = try_all(
        files=[f for f in all_files if f.lower().endswith(".json")],
        match_fn=lambda f: f.lower().endswith(".json"),
        extractor_key="json",
    )
    if geometry:
        return geometry

    # 2. Try all XML files
    geometry = try_all(
        files=[f for f in all_files if f.lower().endswith(".xml")],
        match_fn=lambda f: f.lower().endswith(".xml"),
        extractor_key="xml",
    )
    if geometry:
        return geometry

    # 3. Try Analytic TIFFs
    geometry = try_all(
        files=[f for f in all_files if "_analyticms.tif" in f.lower()],
        match_fn=lambda f: "_analyticms.tif" in f.lower(),
        extractor_key="analytic",
    )
    if geometry:
        return geometry

    # 4. Try UDM TIFFs
    geometry = try_all(
        files=[f for f in all_files if "_udm2.tif" in f.lower()],
        match_fn=lambda f: "_udm2.tif" in f.lower(),
        extractor_key="udm",
    )
    if geometry:
        return geometry

    print("[WARNING] No valid geometry could be extracted from any file.")
    return {}


async def process_roi_mode(
    processor,
    id_dict,
    roi_name,
    shared_geometry_path=None,
    extractors=None,
    order_name=None,
):
    if not extractors:
        raise ValueError("No extractors provided for ROI mode.")

    if shared_geometry_path:
        shared_geometry = extract_geometry_from_geojson(shared_geometry_path)
    else:
        shared_geometry = extract_first_available_geometry(id_dict, extractors)

    if not shared_geometry:
        raise ValueError("No geometry found for ROI.")

    # Insert one ROI into ROIs table
    await processor.process(
        {
            "action": "insert_roi",
            "roi_id": roi_name,
            "tile_id": "",
            "capture_time": "",
            "geometry": shared_geometry,
            "order_name": order_name,
        }
    )

    # Each ROI intersects with multiple tiles so insert each tile id that the ROI intersects with into the DB
    for tile_id, details in id_dict.items():
        # for each tile make a new entry in the roi_tiles table
        await processor.process(
            {
                "action": "insert_roi_tile",
                "roi_id": roi_name,
                "tile_id": tile_id,
                "capture_time": extract_unique_datetime_ids(tile_id),
                "intersection": None,
                "fallback_geom": shared_geometry,
            }
        )

        # Now insert the locations of the files downloaded for that ROI to the database
        for filepath in details["files"]:
            filename = os.path.basename(filepath)
            await processor.process(
                {
                    "action": "update_metadata_roi",
                    "roi_id": roi_name,
                    "tile_id": "_".join(filename.split("_")[:4]),
                    "order_id": "",
                    "filepath": filepath,
                    "status": TILE_STATUSES["DOWNLOADED"],
                }
            )

    # Associate all files with the ROI ID
    for tile_id, details in id_dict.items():
        # for each tile make a new entry in the roi_tiles table
        await processor.process(
            {
                "action": "insert_roi_tile",
                "roi_id": roi_name,
                "tile_id": tile_id,
                "capture_time": extract_unique_datetime_ids(tile_id),
                "intersection": shared_geometry,
                "fallback_geom": shared_geometry,
            }
        )
        for filepath in details["files"]:
            filename = os.path.basename(filepath)
            await processor.process(
                {
                    "action": "update_metadata_roi",
                    "roi_id": roi_name,
                    "tile_id": "_".join(filename.split("_")[:4]),
                    "order_id": "",
                    "filepath": filepath,
                    "status": TILE_STATUSES["DOWNLOADED"],
                }
            )

    print(f"✅ Inserted ROI '{roi_name}' with shared geometry and associated files.")


def extract_id(filename):
    """
    Extracts the ID from the filename by:
      - Removing the '_metadata.json' suffix if present.
      - Splitting the resulting string by underscores.
      - Validating that the first two parts correspond to an 8-digit date and a 6-digit time.
      - Joining the first four segments to form the ID.

    Returns None if the filename does not follow the expected pattern.
    """
    # Remove the metadata suffix if present
    base = filename.replace("_metadata.json", "")
    parts = base.split("_")

    # Ensure there are at least 4 parts
    if len(parts) < 4:
        return None

    # Check if the first part is an 8-digit date and the second is a 6-digit time
    if not (parts[0].isdigit() and len(parts[0]) == 8):
        return None
    if not (parts[1].isdigit() and len(parts[1]) == 6):
        return None

    return "_".join(parts[:4])


def existing_files_dictionary(directory):
    id_dict = {}

    # Walk through the directory tree recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_id = extract_id(file)
            if not file_id:
                continue  # Skip files that don't have an ID

            full_path = os.path.join(root, file)

            # If the ID is new, initialize its dictionary entry
            if file_id not in id_dict:
                id_dict[file_id] = {"metadata": None, "files": []}

            # If the file is a metadata file, record it under "metadata"
            if file.endswith("_metadata.json"):
                id_dict[file_id]["metadata"] = full_path

            # Append every file that has this ID to the list
            id_dict[file_id]["files"].append(full_path)

    return id_dict


async def process_directory(
    directory,
    processor,
    mode="tile",  # or "roi"
    roi_name=None,
    shared_geometry_path=None,
    extractors=None,
    order_name=None,
):

    id_dict = existing_files_dictionary(directory)

    with open("id_mapping.json", "w") as json_file:
        json.dump(id_dict, json_file, indent=4)

    if mode == "roi":
        if not roi_name:
            raise ValueError("ROI mode requires a `roi_name` to be provided.")
        await process_roi_mode(
            processor,
            id_dict,
            roi_name,
            shared_geometry_path,
            extractors,
            order_name=order_name,
        )
    elif mode == "tile":
        await process_tile_mode(processor, id_dict, extractors, order_name=order_name)
    else:
        raise ValueError("Unknown mode. Must be 'tile' or 'roi'.")


def find_file(directory, target_filename):
    """
    Recursively search through a directory for a specific file.

    Args:
        directory (str): The root directory to start the search.
        target_filename (str): The name of the file to search for (e.g., 'manifest.json').

    Returns:
        str: Full path to the file if found, otherwise an empty string.
    """
    for root, _, files in os.walk(directory):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return ""


def is_ROI_folder(folder_path) -> bool:
    """
    Check if a folder name is a valid ROI folder.
    This is a placeholder function and should be implemented based on actual criteria.
    """
    found = search_files(folder_path, r".*\.(tiff?|xml)$", ["clip", "toar"])
    return found


async def process_all_subfolders(parent_directory, processor, extractors):
    """
    Process each folder in the parent directory using either ROI or Tile mode,
    depending on whether it qualifies as an ROI folder.

    Args:
        parent_directory (str): The path to the parent directory containing subfolders.
        processor: The TileProcessor instance used to interact with the database.
        extractors (dict): A dictionary of extractor functions.
            These functions are used to extract geometry from various file types.
                Example: {"json": lambda path: read_json(path).get("geometry", {})}
    """
    for entry in os.scandir(parent_directory):
        if entry.is_dir():
            folder_path = entry.path
            folder_name = os.path.basename(folder_path)

            # Insert order name into the database
            insert_order(folder_path, processor)
            print(f"\n🔍 Checking folder: {folder_name}")

            if is_ROI_folder(folder_path):
                print(f"📁 Detected ROI folder: {folder_name}")
                await process_directory(
                    directory=folder_path,
                    processor=processor,
                    mode="roi",
                    extractors=extractors,
                    roi_name=folder_name,
                    order_name=folder_name,
                )
            else:
                print(f"📁 Detected TILE folder: {folder_name}")
                await process_directory(
                    directory=folder_path,
                    processor=processor,
                    mode="tile",
                    extractors=extractors,
                    order_name=folder_name,
                )

            print(f"✅ Finished processing folder: {folder_name}")


async def add_existing_orders_to_db(directory, extractors):
    """
    Process all subfolders in the given directory and add existing orders to the database.

    Args:
        directory (str): The path to the parent directory containing subfolders.
        extractors (dict): A dictionary of extractor functions.
            These functions are used to extract geometry from various file types.
                Example: {"json": lambda path: read_json(path).get("geometry", {})}
    """
    processor = create_processor(DATABASE_PATH)
    await process_all_subfolders(directory, processor, extractors)


def create_processor(db_path):
    """
    Initializes the database and returns a TileProcessor instance.
    """
    db = BaseDuckDB(db_path)
    db.use_spatial_extension()
    db.create_tables()

    roi_repo = ROIRepository(db)
    tile_repo = TileRepository(db)
    order_repo = OrderRepository(db)

    return TileProcessor(roi_repo, tile_repo, order_repo)
