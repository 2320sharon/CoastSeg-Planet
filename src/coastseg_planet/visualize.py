import imageio
import numpy as np
import os
from PIL import Image

import rasterio
from ipyleaflet import Map, ImageOverlay
from ipywidgets import VBox, FloatRangeSlider


import numpy as np
import PIL.Image
from io import BytesIO
import base64

# Function to create a base64 PNG image from mask data with transparency
def create_masked_image(mask_data, alpha=128):
    # Normalize the mask data to 0-255
    mask_data = (mask_data / mask_data.max() * 255).astype(np.uint8)
    # Create an RGBA image with the red channel filled and alpha channel set by the mask
    rgba_image = np.zeros((mask_data.shape[0], mask_data.shape[1], 4), dtype=np.uint8)
    rgba_image[..., 0] = mask_data  # Red channel
    # Set alpha to the specified value for the masked regions, 0 for the unmasked regions
    rgba_image[..., 3] = np.where(mask_data > 0, alpha, 0)  # Alpha channel (transparency)
    img = PIL.Image.fromarray(rgba_image)
    with BytesIO() as buffer:
        img.save(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def create_interactive_map(topobathy_tiff):
    with rasterio.open(topobathy_tiff) as src:
        mask_data = src.read(1)  # Read the first band
        bounds = src.bounds

    # Create the initial masked image overlay
    mask_data = np.flipud(mask_data)  # Flip the mask data vertically
    masked_image = create_masked_image(mask_data)
    image_overlay = ImageOverlay(
        url="data:image/png;base64," + masked_image,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    )

    # Create the map
    m = Map(center=[(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2], zoom=13)
    m.add_layer(image_overlay)

    # Function to update the mask (simulated here for demonstration purposes)
    def update_mask(change):
        if not isinstance(change, tuple):
            mask = mask_data < change
            masked_image = create_masked_image(mask)
            image_overlay.url = "data:image/png;base64," + masked_image
        else:
            lower, upper = change
            mask = np.logical_and(mask_data > lower, mask_data < upper)
            masked_image = create_masked_image(mask)
            image_overlay.url = "data:image/png;base64," + masked_image

    default_value = (-10, 10)
    # Create a bounded slider with two handles
    threshold_slider = FloatRangeSlider(
        description='Elevation (m)',
        value=[default_value[0], default_value[1]],
        min=mask_data.min(),
        max=mask_data.max(),
        step=0.1,
    )

    # Link the slider to the update function
    threshold_slider.observe(lambda change: update_mask(change['new']), names='value')

    # Display the map and the slider

    vbox = VBox([m, threshold_slider])

    # Initial display
    update_mask(default_value)

    # Functions to access slider values
    def get_low_threshold():
        return threshold_slider.value[0]

    def get_high_threshold():
        return threshold_slider.value[1]

    return vbox, get_low_threshold, get_high_threshold

def find_max_dimensions(tiff_files):
    max_height, max_width = 0, 0
    for tiff_file in tiff_files:
        try:
            img = imageio.imread(tiff_file)
            height, width = img.shape[:2]
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        except:
            print(f"Could not read {tiff_file}")
            continue
    return max_height, max_width

def pad_to_max_dimensions(image, max_height, max_width):
    height, width = image.shape[:2]
    pad_height = max_height - height
    pad_width = max_width - width

    # Calculate padding needed to maintain aspect ratio
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    return np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

def resize_to_divisible_by_16(image):
    height, width = image.shape[:2]
    new_height = (height + 15) // 16 * 16
    new_width = (width + 15) // 16 * 16
    pad_height = new_height - height
    pad_width = new_width - width

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    return np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')

def create_movie_from_tiffs(tiff_files, output_movie, fps=10):
    writer = imageio.get_writer(output_movie, fps=fps, format='FFMPEG')

    max_height, max_width = find_max_dimensions(tiff_files)

    for tiff_file in tiff_files:
        try:
            img = imageio.imread(tiff_file)
        except:
            print(f"Could not read {tiff_file}")
            continue
        img_padded = pad_to_max_dimensions(img, max_height, max_width)
        img_resized = resize_to_divisible_by_16(img_padded)
        writer.append_data(img_resized)

    writer.close()
    print(f"Movie saved to {os.path.abspath(output_movie)}")

def convert_tiffs_to_jpgs(tiff_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jpg_files = []
    for tiff_file in tiff_files:
        try:
            img = imageio.v2.imread(tiff_file)

            # Extract the first 3 bands (assuming they are B, G, R)
            img_bgr = img[:, :, :3]

            # Normalize the values to the range [0, 255]
            img_bgr = img_bgr.astype(np.float32)
            img_bgr -= img_bgr.min()
            img_bgr /= img_bgr.max()
            img_bgr *= 255.0
            
            # Convert to 8-bit unsigned integer format
            img_bgr_8bit = img_bgr.astype(np.uint8)
            
            # Convert BGR to RGB
            img_rgb = img_bgr_8bit[:, :, ::-1]
            
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            jpg_file = os.path.join(output_dir, os.path.basename(tiff_file).replace('.tif', '.jpg'))
            
            img_pil.save(jpg_file, 'JPEG')

            jpg_files.append(jpg_file)
        except Exception as e:
            print(f"Could not convert {tiff_file}: {e}")
            continue

    return jpg_files

def create_movie_from_images(image_files, output_movie, fps=10):
    writer = imageio.get_writer(output_movie, fps=fps, format='FFMPEG')

    max_height, max_width = find_max_dimensions(image_files)

    for image_file in image_files:
        try:
            img = imageio.imread(image_file)
        except:
            print(f"Could not read {image_file}")
            continue
        img_padded = pad_to_max_dimensions(img, max_height, max_width)
        img_resized = resize_to_divisible_by_16(img_padded)
        writer.append_data(img_resized)

    writer.close()
    print(f"Movie saved to {os.path.abspath(output_movie)}")

# import imageio
# import numpy as np
# import os

# def find_max_dimensions(tiff_files):
#     max_height, max_width = 0, 0
#     for tiff_file in tiff_files:
#         try:
#             img = imageio.imread(tiff_file)
#             height, width = img.shape[:2]
#             if height > max_height:
#                 max_height = height
#             if width > max_width:
#                 max_width = width
#         except:
#             print(f"Could not read {tiff_file}")
#             continue
#     return max_height, max_width

# def resize_to_max_dimensions(image, max_height, max_width):
#     height, width = image.shape[:2]
#     pad_height = max_height - height
#     pad_width = max_width - width
#     return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

# def resize_to_divisible_by_16(image):
#     height, width = image.shape[:2]
#     new_height = (height + 15) // 16 * 16
#     new_width = (width + 15) // 16 * 16
#     return np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), mode='constant')

# def create_movie_from_tiffs(tiff_files, output_movie, fps=10):
#     writer = imageio.get_writer(output_movie, fps=fps,format='FFMPEG')

#     max_height, max_width = find_max_dimensions(tiff_files)

#     for tiff_file in tiff_files:
#         try:
#             img = imageio.imread(tiff_file)
#         except:
#             print(f"Could not read {tiff_file}")
#             continue
#         img_resized = resize_to_max_dimensions(img, max_height, max_width)
#         img_resized = resize_to_divisible_by_16(img_resized)
#         writer.append_data(img_resized)

#     writer.close()
#     print(f"Movie saved to {os.path.abspath(output_movie)}")

    
# # # # specify the pattern to match your tiff files
# # # good_dir = r"C:\development\coastseg-planet\CoastSeg-Planet\santa_cruz_boardwalk_QGIS\files\good"
# # # tiff_files = sorted(glob.glob(os.path.join(good_dir, f"*3B_TOAR_processed_coregistered.tif")))
# # # output_movie = 'coregistered_santa_cruz_planet2.mp4'

# # # create_movie_from_tiffs(tiff_files, output_movie)