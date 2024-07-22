import imageio
import numpy as np
import glob
import os

def find_max_dimensions(tiff_files):
    max_height, max_width = 0, 0
    for tiff_file in tiff_files:
        img = imageio.imread(tiff_file)
        height, width = img.shape[:2]
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
    return max_height, max_width

def resize_to_max_dimensions(image, max_height, max_width):
    height, width = image.shape[:2]
    pad_height = max_height - height
    pad_width = max_width - width
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

def resize_to_divisible_by_16(image):
    height, width = image.shape[:2]
    new_height = (height + 15) // 16 * 16
    new_width = (width + 15) // 16 * 16
    return np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), mode='constant')

def create_movie_from_tiffs(tiff_files, output_movie, fps=10):
    writer = imageio.get_writer(output_movie, fps=fps)

    max_height, max_width = find_max_dimensions(tiff_files)

    for tiff_file in tiff_files:
        img = imageio.imread(tiff_file)
        img_resized = resize_to_max_dimensions(img, max_height, max_width)
        img_resized = resize_to_divisible_by_16(img_resized)
        writer.append_data(img_resized)

    writer.close()
    print(f"Movie saved to {os.path.abspath(output_movie)}")

# specify the pattern to match your tiff files
# good_dir = r"C:\development\coastseg-planet\CoastSeg-Planet\santa_cruz_boardwalk_QGIS\files\good"
good_dir = r"C:\development\coastseg-planet\downloads\Alaska_TOAR_enabled\42444c87-d914-4330-ba4b-3c948829db3f\PSScene\good"

tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*3B_TOAR_processed_coregistered_global.tif")))
output_movie = 'coregistered_AK_global_v1.mp4'
create_movie_from_tiffs(tiff_files, output_movie)

# do it again but use the original tiff before coregistering
# specify the pattern to match your tiff files
tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*_3B_TOAR_model_format.tif")))
output_movie = 'original_AK_v1.mp4'
create_movie_from_tiffs(tiff_files, output_movie)



# specify the pattern to match your tiff files
tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*_3B_TOAR_processed_coregistered_local.tif")))
output_movie = 'coregistered_AK_local_v1.mp4'
create_movie_from_tiffs(tiff_files, output_movie)
