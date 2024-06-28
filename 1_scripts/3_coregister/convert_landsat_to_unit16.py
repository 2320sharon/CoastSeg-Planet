import rasterio
import numpy as np

def process_raster(input_path, output_path):
    with rasterio.open(input_path) as input_raster:
        out_meta = input_raster.meta.copy()
        modified_data = np.empty((input_raster.count, input_raster.height, input_raster.width), dtype=out_meta['dtype'])
        scale = 10000
        for i in range(1, input_raster.count + 1):
            data = input_raster.read(i)
            data[np.isinf(data)] = 0
            data = data * scale
            modified_data[i-1] = data

        out_meta.update(
            dtype=rasterio.uint16,
            count=input_raster.count,
            nodata=0
        )

    print(f"output_path: {output_path}")
    with rasterio.open(output_path, 'w', **out_meta) as output_raster:
        output_raster.write(modified_data)
        
import os   
# Enter the tif location of of the landsat TOA reflectance image here     
# input_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_mqq51_datetime05-06-24__06_47_51\L8\ms\2023-02-14-10-33-43_L8_ID_mqq51_datetime05-06-24__06_47_51_ms.tif"  

input_path = r"C:\development\coastseg-planet\CoastSeg-Planet\2024-01-24-10-33-39_L9_ID_mqq51_datetime05-06-24__06_47_51_ms_new_block.tif"
# Enter the full path of the output tif file here
# output_path = '2023-02-14-10-33-43_L8_ID_mqq51_unit16.tif'
output_path = os.path.basename(input_path).replace('.tif', '_unit16.tif')
# convert the TOA reflectance image to uint16 by multiplying by 10000
process_raster(input_path, output_path)

# print the metadata of the output tif file
with rasterio.open(output_path) as new_tif:
    print(new_tif.indexes)
    for i, dtype, nodataval in zip(new_tif.indexes, new_tif.dtypes, new_tif.nodatavals):
        print(i, dtype, nodataval)
        print(f"Min: {np.min(new_tif.read(i))}")
        print(f"Max: {np.max(new_tif.read(i))}")