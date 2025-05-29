import os
from typing import Optional
import rasterio
from rasterio.mask import mask
import numpy as np
from xml.dom import minidom
from shapely.geometry import shape, mapping
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling


def clip_geotiff_to_roi(
    geotiff_path: str, roi_gdf: gpd.GeoDataFrame, output_path: Optional[str] = None
) -> str:
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


def convert_to_toa(
    image_path: str, xml_path: str, save_path: str, output_epsg: Optional[str] = None
) -> str:
    with rasterio.open(image_path) as src:
        bands = src.read()
        meta = src.meta.copy()

    xmldoc = minidom.parse(xml_path)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    coeffs = {
        int(node.getElementsByTagName("ps:bandNumber")[0].firstChild.data): float(
            node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
        )
        for node in nodes
        if node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        in ["1", "2", "3", "4"]
    }

    scale = 10000
    toa_scaled = np.array([bands[i] * coeffs[i + 1] * scale for i in range(4)]).astype(
        rasterio.uint16
    )

    sensor = (
        xmldoc.getElementsByTagName("eop:Instrument")[0]
        .getElementsByTagName("eop:shortName")[0]
        .firstChild.data
    )
    suffix_map = {
        "PS2": "_PS2_TOA.tif",
        "PS2.SD": "_2SD_TOA.tif",
        "PSB.SD": "_BSD_TOA.tif",
    }
    save_path += suffix_map.get(sensor, "_UNKNOWN_TOA.tif")

    meta.update(dtype=rasterio.uint16, count=4, compress="lzw")
    with rasterio.open(save_path, "w", **meta) as dst:
        for i in range(4):
            dst.write(toa_scaled[i], i + 1)

    if output_epsg:
        current_crs = str(meta["crs"]).replace("epsg", "EPSG")
        if current_crs != output_epsg:
            save_path = raster_change_epsg(
                {"output_epsg": output_epsg}, save_path, "0", current_crs
            )

    return save_path


def raster_change_epsg(output_epsg_dict, raster_path, suffix="", src_crs_str=None):
    output_epsg = output_epsg_dict["output_epsg"]
    out_path = raster_path.replace(".tif", f"_reproj{suffix}.tif")

    with rasterio.open(raster_path) as src:
        src_crs = (
            src.crs
            if src_crs_str is None
            else rasterio.crs.CRS.from_string(src_crs_str)
        )
        dst_crs = rasterio.crs.CRS.from_string(output_epsg)

        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
            }
        )

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    return out_path


import rasterio
import numpy as np
from xml.dom import minidom
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape, mapping
import geopandas as gpd
from typing import Optional


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

    return output_path


# === High-level composable pipeline ===
def clip_and_convert_to_toa(
    input_tiff: str, xml_path: str, roi_path: str, output_epsg: Optional[str] = None
) -> str:
    roi_gdf = gpd.read_file(roi_path)
    clipped_tiff = clip_geotiff_to_roi(input_tiff, roi_gdf)
    final_tiff = convert_to_toa(
        clipped_tiff,
        xml_path,
        clipped_tiff.replace(".tif", ""),
        output_epsg=output_epsg,
    )
    return final_tiff


input_tiff = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\flying_spit_roi_437_2024-09-16_2024-09-30\defc45cb-d940-4ac6-a759-240af618aad2\PSScene\20240924_222725_49_24e5_3B_AnalyticMS.tif"
xml_path = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\flying_spit_roi_437_2024-09-16_2024-09-30\defc45cb-d940-4ac6-a759-240af618aad2\PSScene\20240924_222725_49_24e5_3B_AnalyticMS_metadata.xml"
save_path = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\flying_spit_roi_437_2024-09-16_2024-09-30\defc45cb-d940-4ac6-a759-240af618aad2\PSScene\20240924_222725_49_24e5_3B_AnalyticMS_TOA.tif"

ROI_NUMBER = 437
roi_path = f"C:/development/1_coastseg_planet/1_coastseg-planet/CoastSeg-Planet/3_demo_data/flying_spit_roi_{ROI_NUMBER}.geojson"
output_path = r"C:\development\1_coastseg_planet\CoastSeg-Planet\downloads\flying_spit_roi_437_2024-09-16_2024-09-30\defc45cb-d940-4ac6-a759-240af618aad2\PSScene\20240924_222725_49_24e5_3B_AnalyticMS_clip.tif"

# this function creates intermediate files
# final_path = clip_and_convert_to_toa(
#     input_tiff=input_tiff,
#     xml_path=xml_path,
#     roi_path=roi_path,
# )
# print(f"Final TOA TIFF saved to: {final_path}")


output_tif = clip_toa_reproject_geotiff(
    image_path=input_tiff,
    xml_path=xml_path,
    roi_path=roi_path,
    output_path=input_tiff.replace(".tif", "_clip_TOAR.tif"),
)
print("Saved:", output_tif)
