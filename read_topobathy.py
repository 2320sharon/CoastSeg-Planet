
## Downloads topobathy data using https://github.com/NeptuneProjects/BathyReq
## and makes a pretty plot
## written by Dr Daniel Buscombe, April 26-30, 2024

## Example usage, from cmd:
# python download_topobathymap_geojson.py -s "LongIsland" -f "/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/longisland_example.geojson"

import os
import bathyreq
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib.colors
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
import rasterio.warp

input_file=r"C:\development\coastseg-planet\CoastSeg-Planet\Santa Cruz Boardwalk_topobathy.tif"

with rasterio.open(input_file)as r:
    print(r.profile)
    band1=r.read(1)
    # it will be upside down, so flip it (why is this??)
    band1 = np.flipud(band1)
    print(band1.shape)
    # print(band1)
    elevation_threshold_high = 5
    elevation_threshold_low = -5
    mask = np.logical_and(band1 > elevation_threshold_low, band1 < elevation_threshold_high)
    # mask = band1<= elevation_threshold
    # Plot the original band
    plt.imshow(band1, cmap='gray')
    
    # Overlay the mask with transparency
    plt.imshow(mask, cmap='jet', alpha=0.4)
    # Create a legend for the mask
    legend_elements = [Patch(facecolor='r', edgecolor='r',
                            label=f'True ( {elevation_threshold_high} >= band1 > {elevation_threshold_low})'),
                    Patch(facecolor='b', edgecolor='b',
                            label=f'False (band1> {elevation_threshold_high} or band1 <= {elevation_threshold_low})')]
    # legend_elements = [Patch(facecolor='r', edgecolor='r',
    #                         label=f'True (band1<= {elevation_threshold})'),
    #                 Patch(facecolor='b', edgecolor='b',
    #                         label=f'False (band1> {elevation_threshold})')]

    plt.legend(handles=legend_elements, loc='upper right')
    file_name = f"mask_elevation_threshold_{elevation_threshold_high}_{elevation_threshold_low}.png"
    print(os.path.abspath(file_name))
    plt.savefig(file_name)
    plt.show()
    # Use the mask to index band1
    values_greater_than_10 = band1[mask]
    
    print(values_greater_than_10)

