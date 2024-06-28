# Standard library imports
import os

os.environ["TF_USE_LEGACY_KERAS"] = "False" # this is needed in order for the classification model to run 

import json
from typing import List
import glob

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.transform import resize
from tensorflow import keras

# Local application imports
from coastseg_planet.processing import read_tiff
from doodleverse_utils.prediction_imports import est_label_binary, est_label_multiclass, label_to_colors
from doodleverse_utils.model_imports import dice_coef_loss
from doodleverse_utils.model_imports import segformer
from coastseg_planet.plotting import plot_overlay, plot_side_by_side_overlay, plot_per_class_probabilities
from coastseg_planet.utils import sort_images, resize_array
from doodleverse_utils.imports import standardize, label_to_colors

# GLOBAL VARIABLES
CLASS_LABEL_COLORMAPS = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
        "#316395",
        "#ffe4e1",
        "#ff7373",
        "#666666",
        "#c0c0c0",
        "#66cdaa",
        "#afeeee",
        "#0e2f44",
        "#420420",
        "#794044",
        "#3399ff",
    ]

def apply_model_to_dir(directory: str, suffix: str):
    """
    Apply a model to all images in a directory with a specific suffix.

    Args:
        directory (str): The directory path where the images are located.
        suffix (str): The suffix of the images to be processed.

    Returns:
        None
    """
    for target_path in glob.glob(os.path.join(directory, f"*{suffix}.tif")):
        apply_model_to_image(target_path, directory, False, False)


def read_file_for_classification_model(file_path:str,length:int=128, width:int=128,bands:int=3)->np.ndarray:
    """
    Reads an image file and returns it as a NumPy array, s it can be read by the model.

    Parameters:
    file_path (str): The path to the image file.

    Returns:
    np.ndarray: The image data as a NumPy array.

    Raises:
    ValueError: If the file type is not supported.
    """
    if file_path.endswith(".tif") or file_path.endswith(".tiff"):
        img_array = read_tiff(file_path)
        img_array = resize_array(img_array,(length,width,bands))
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    elif file_path.split('.')[-1] in ['jpg','jpeg','png']:
        img = keras.utils.load_img(file_path, target_size=(length,width))
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    else:
        raise ValueError(f"File type not supported: {file_path}")

def run_classification_model(path_to_model:str,
                  path_to_inference_imgs,
                  output_folder,
                  csv_path:str,
                  regex_pattern='',
                  move_files:bool=True,):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    csv_path (str): csv path to save results to
    move_files (bool): if True, move the images to the good and bad folders, if False, copy the images
    returns:
    csv_path (str): csv path of saved results
    """
    model = keras.models.load_model(path_to_model)
    # get the image paths that end in jpg, jpeg, png or tif
    file_types = ('*.jpg', '*.jpeg', '*.png','*tif') # the tuple of file types
    im_paths = []
    for file_extension in file_types:
        if regex_pattern:
            im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, regex_pattern+ file_extension)))
        else:
            im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, file_extension)))
    # print(f"im_paths: {im_paths}")
    im_classes = [None]*len(im_paths)
    im_scores = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img_array = read_file_for_classification_model(im_path,128,128,3)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        good_score = score
        bad_score = 1 - score
        if good_score>bad_score:
            im_classes[i] = 'good'
            im_scores[i] = good_score
        else:
            im_classes[i] = 'bad'
            im_scores[i] = bad_score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'im_classes':im_classes,
                       'im_scores':im_scores})
    df.to_csv(csv_path)
    sort_images(csv_path,
                output_folder,move=move_files)
    return csv_path

# def read_file_for_model(file_path:str)->np.ndarray:
#     """
#     Reads an image file and returns it as a NumPy array, s it can be read by the model.

#     Parameters:
#     file_path (str): The path to the image file.

#     Returns:
#     np.ndarray: The image data as a NumPy array.

#     Raises:
#     ValueError: If the file type is not supported.
#     """
#     if file_path.endswith(".tif"):
#         return read_tiff(file_path)
#     elif file_path.endswith(".jpg") or file_path.endswith(".png"):
#         return imread(file_path)
#     else:
#         raise ValueError(f"File type not supported: {file_path}")

def apply_model_to_image(input_image:str, save_path:str, TESTTIMEAUG:bool=False, OTSU_THRESHOLD:bool=False):
    """
    Applies a trained model to an input image for segmentation.
    
    Input file must be a 4 band tiff image with band order of RGBNIR with values between 0 and 255.

    Args:
        input_image (str): The path to the input image file.
        save_path (str): The path to the directory where the output image will be saved.
        TESTTIMEAUG (bool, optional): Whether to apply test-time augmentation. Defaults to False.
        OTSU_THRESHOLD (bool, optional): Whether to use Otsu thresholding. Defaults to False.

    Returns:
        None
    """
    
    # check the range of the image to be sure it is between 0 and 255
    
    # directory to save outputs to temporarily hard coded for now
    # sample_direc = r'C:\development\coastseg-planet\CoastSeg-Planet\output_zoo'
    # hard code this for now.... segformer only
    weights_directory = r'C:\development\doodleverse\coastseg\CoastSeg\src\coastseg\downloaded_models\segformer_RGB_4class_8190958'
    # get the model weights needed to initialize the model
    weights_list = get_weights_list(weights_directory)
    # load the model
    model, model_list, config_files, model_types = get_model(weights_list)
    # load the dictionary of the configuration options for the model
    config, config_file = get_config(weights_list)
    
    metadata_dict = get_metadatadict(
            weights_list, config_files, model_types
        )
    do_seg(
        input_image,
        model_list,
        metadata_dict,
        model_types[0],
        sample_direc=save_path,
        NCLASSES=config.get('NCLASSES'),
        N_DATA_BANDS=config.get('N_DATA_BANDS'),
        TARGET_SIZE=config.get('TARGET_SIZE'),
        TESTTIMEAUG=TESTTIMEAUG,
        WRITE_MODELMETADATA=True, # this controls whether the npz file is written 
        OTSU_THRESHOLD=OTSU_THRESHOLD,
        profile="meta",
    )

def seg_file2tensor_3band(f, TARGET_SIZE):  
    """
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bigimage = imread(f) 
    # if a 4 band image, discard the 4th band ( typically NIR)
    if bigimage.shape[-1]==4:
        bigimage = bigimage[:,:,:3] 
        
    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage



def get_metadatadict(
    weights_list: list, config_files: list, model_types: list
) -> dict:
    """
    Returns a dictionary containing metadata information.

    Args:
        weights_list (list): A list of model weights.
        config_files (list): A list of configuration files.
        model_types (list): A list of model types.

    Returns:
        dict: A dictionary containing the metadata information.
    """
    metadatadict = {}
    metadatadict["model_weights"] = weights_list
    metadatadict["config_files"] = config_files
    metadatadict["model_types"] = model_types
    return metadatadict

def get_config(weights_list: list):
    """
    Retrieves the configuration and configuration file path based on the provided weights list.

    Args:
        weights_list (list): A list of weights.

    Returns:
        tuple: A tuple containing the configuration (dict) and the configuration file path (str).
        
    Raises:
        Exception: If no model info is passed (weights_list is empty).
    """
    if weights_list == []:
        raise Exception("No Model Info Passed")
    for weights in weights_list:
        weights = weights.strip()
        config_file = weights.replace(".h5", ".json").replace("weights", "config")
        if "fullmodel" in config_file:
            config_file = config_file.replace("_fullmodel", "")
        with open(config_file) as f:
            config = json.load(f)
    return config, config_file

def get_model(weights_list: list):
    """
    Loads and returns a list of models based on the provided weights list.

    Args:
        weights_list (list): A list of paths to the model weights files.

    Returns:
        tuple: A tuple containing the following elements:
            - model: The last loaded model from the weights list.
            - model_list (list): A list of all loaded models.
            - config_files (list): A list of configuration files associated with each model.
            - model_types (list): A list of model types corresponding to each loaded model.
    """
    model_list = []
    config_files = []
    model_types = []
    config, config_file = get_config(weights_list)
    for weights in weights_list:
        try:
            model = tf.keras.models.load_model(weights)
        except BaseException:
            if config.get("MODEL") == "segformer":
                id2label = {}
                for k in range(config.get("NCLASSES")):
                    id2label[k] = str(k)
                model = segformer(id2label, num_classes=config.get("NCLASSES"))
                model.compile(optimizer="adam")
            model.compile(
                optimizer="adam", loss=dice_coef_loss(config.get("NCLASSES"))
            ) 
            model.load_weights(weights)
        model_types.append(config.get("MODEL"))
        model_list.append(model)
        config_files.append(config_file)
    return model, model_list, config_files, model_types

def get_weights_list(weights_directory, model_choice: str = "BEST") -> List[str]:
    """Returns a list of the model weights files (.h5) within the weights directory.
    Args:
        model_choice (str, optional): The type of model weights to return.
            Valid choices are 'ENSEMBLE' (default) to return all available
            weights files or 'BEST' to return only the best model weights file.
    Returns:
        list: A list of strings representing the file paths to the model weights
        files in the weights directory.
    Raises:
        FileNotFoundError: If the BEST_MODEL.txt file is not found in the weights directory.
    Example:
        trainer = ModelTrainer(weights_direc='/path/to/weights')
        weights_list = trainer.get_weights_list(model_choice='ENSEMBLE')
        print(weights_list)
        # Output: ['/path/to/weights/model1.h5', '/path/to/weights/model2.h5', ...]
        best_weights_list = trainer.get_weights_list(model_choice='BEST')
        print(best_weights_list)
        # Output: ['/path/to/weights/best_model.h5']
    """
    if model_choice == "ENSEMBLE":
        weights_list = glob(os.path.join(weights_directory, "*.h5"))
        return weights_list
    elif model_choice == "BEST":
        # read model name (fullmodel.h5) from BEST_MODEL.txt
        with open(os.path.join(weights_directory, "BEST_MODEL.txt")) as f:
            model_name = f.readline()
        # remove any leading or trailing whitespace and newline characters
        model_name = model_name.strip()
        weights_list = [os.path.join(weights_directory, model_name)]
        return weights_list
    else:
        raise ValueError(
            f"Invalid model_choice: {model_choice}. Valid choices are 'ENSEMBLE' or 'BEST'."
        )


def seg_file2tensor_ND(f, TARGET_SIZE):  
    """
    "seg_file2tensor(f)"
    This function reads a NPZ image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of npz
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
    GLOBAL INPUTS: TARGET_SIZE
    """

    with np.load(f) as data:
        bigimage = data["arr_0"].astype("uint8")

    smallimage = resize(
        bigimage, (TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True
    )
    smallimage = np.array(smallimage)
    smallimage = tf.cast(smallimage, tf.uint8)

    w = tf.shape(bigimage)[0]
    h = tf.shape(bigimage)[1]

    return smallimage, w, h, bigimage

# #-----------------------------------
def get_image(file_path:str, N_DATA_BANDS:int, TARGET_SIZE:int, MODEL:str):
    """
    Retrieves an image and its associated information based on the number of data bands.

    Args:
        file_path (str): The file path of the image.
        N_DATA_BANDS (int): The number of data bands in the image.
        TARGET_SIZE (int): The target size of the image.
        MODEL (str): The model type.

    Returns:
        tuple: A tuple containing the image, width, height, and big image.

    Raises:
        None
    """
    print(f"N_DATA_BANDS: {N_DATA_BANDS}")
    if N_DATA_BANDS <= 3:
        image, w, h, bigimage = seg_file2tensor_3band(file_path, TARGET_SIZE)
    else:
        image, w, h, bigimage = seg_file2tensor_ND(file_path, TARGET_SIZE)

    try: #>3 bands
        if N_DATA_BANDS<=3:
            if image.shape[-1]>3:
                image = image[:,:,:3]

            if bigimage.shape[-1]>3:
                bigimage = bigimage[:,:,:3]
    except:
        pass

    image = standardize(image.numpy()).squeeze()

    if MODEL=='segformer':
        if np.ndim(image)==2:
            image = np.dstack((image, image, image))
        image = tf.transpose(image, (2, 0, 1))

    return image, w, h, bigimage 

def process_image(filename, N_DATA_BANDS, TARGET_SIZE, MODEL):
    """
    Process an image by loading it, resizing it, and checking if it is empty.

    Args:
        filename (str): The path to the image file.
        N_DATA_BANDS (int): The number of data bands in the image.
        TARGET_SIZE (int): The target size for resizing the image.
        MODEL: The model used for processing the image.

    Returns:
        tuple: A tuple containing the processed image, width, height, and the original image.

    """
    image, w, h, bigimage = get_image(filename, N_DATA_BANDS, TARGET_SIZE, MODEL)
    # if the standard deviation of the image is 0 all the pixels are the same meaning there is nothing of interest in the image (aka empty image)
    if np.std(image) == 0:
        print("Image {} is empty".format(filename))
        return None, np.zeros((w, h)), np.zeros((w, h))
    return image, w, h, bigimage

def calculate_est_label_binary(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h):
    """
    Calculates the estimated label and softmax scores for a binary classification task.

    Args:
        image (numpy.ndarray): The input image.
        M (numpy.ndarray): The segmentation mask.
        MODEL: The model used for prediction.
        TESTTIMEAUG(bool): The test-time augmentation.
        NCLASSES (int): The number of classes.
        TARGET_SIZE: The target size of the image.
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        est_label (numpy.ndarray): The estimated label.
        softmax_scores (numpy.ndarray): The softmax scores.

    """
    # E0 and E1 contain resized predictions for each model, representing the two classes.
    E0, E1 = est_label_binary(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h)
    # Using the predictions from all the models, average the scores such that the estimated label is the average of the all the model scores
    e0 = np.average(np.dstack(E0), axis=-1)
    e1 = np.average(np.dstack(E1), axis=-1)
    # The estimated label is the average of the two classes, which is the probability of class 1
    est_label = (e1 + (1 - e0)) / 2
    # The softmax scores are the two classes
    softmax_scores = np.dstack((e0, e1))
    return est_label, softmax_scores

def apply_threshold(est_label, OTSU_THRESHOLD):
    """
    Applies a threshold to the given estimated label.

    Parameters:
    - est_label: numpy.ndarray
        The estimated label to apply the threshold to.
    - OTSU_THRESHOLD: bool
        If True, the threshold is calculated using Otsu's method.
        If False, a fixed threshold of 0.5 is used.

    Returns:
    - numpy.ndarray
        The thresholded label.
    """
    if OTSU_THRESHOLD:
        thres = threshold_otsu(est_label)
        est_label = (est_label > thres).astype("uint8")
    else:
        est_label = (est_label > 0.5).astype("uint8")
    return est_label

def calculate_est_label_multiclass(image, M, MODEL, TESTTIMEAUG:bool, NCLASSES:int, TARGET_SIZE, w, h):
    """
    Calculates the estimated label for a multiclass image segmentation task.

    Args:
        image (numpy.ndarray): The input image.
        M (numpy.ndarray): The model parameters.
        MODEL (str): The name of the model.
        TESTTIMEAUG (bool): Flag indicating whether to use test-time augmentation.
        NCLASSES (int): The number of classes.
        TARGET_SIZE (tuple): The target size of the image.
        w (int): The width of the output label.
        h (int): The height of the output label.

    Returns:
        tuple: A tuple containing the estimated label and a copy of the estimated label.
    """
    est_label, counter = est_label_multiclass(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE)
    est_label /= counter + 1
    est_label = est_label.numpy().astype('float32')
    if MODEL == 'segformer':
        est_label = resize(est_label, (1, NCLASSES, TARGET_SIZE[0], TARGET_SIZE[1]), preserve_range=True, clip=True).squeeze()
        est_label = np.transpose(est_label, (1, 2, 0))
        est_label = resize(est_label, (w, h))
    else:
        est_label = resize(est_label, (w, h))
    return est_label, est_label.copy()

def save_segmentation_results(segfile, color_label):
    """
    Save the segmentation results to a file.

    Args:
        segfile (str): The path to the file where the segmentation results will be saved.
        color_label (numpy.ndarray): The color labels representing the segmentation results.

    Returns:
        None
    """
    print(f"saving segmentation results to {segfile}")
    imsave(segfile, (color_label).astype(np.uint8), check_contrast=False)

def generate_color_label(est_label, bigimage, class_label_colormap):
    """
    Generates a color label image based on the estimated label, the big image, and the class label colormap.

    Args:
        est_label (numpy.ndarray): The estimated label image.
        bigimage (numpy.ndarray): The big image.
        class_label_colormap (list): The colormap for the class labels.

    Returns:
        numpy.ndarray: The color label image.

    Raises:
        None

    """
    try:
        color_label = label_to_colors(
            est_label,
            bigimage.numpy()[:, :, 0] == 0,
            alpha=128,
            colormap=class_label_colormap,
            color_class_offset=0,
            do_alpha=False,
        )
    except:
        try:
            color_label = label_to_colors(
                est_label,
                bigimage[:, :, 0] == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )
        except:
            color_label = label_to_colors(
                est_label,
                bigimage == 0,
                alpha=128,
                colormap=class_label_colormap,
                color_class_offset=0,
                do_alpha=False,
            )        
    return color_label

def get_segfile_name(filename):
    if filename.endswith("jpg"):
        segfile = filename.replace(".jpg", "_predseg.png")
    elif filename.endswith("png"):
        segfile = filename.replace(".png", "_predseg.png")
    elif filename.endswith("npz"):  # in filename:
        segfile = filename.replace(".npz", "_predseg.png")
    elif filename.endswith("tif"):  # in filename:
        segfile = filename.replace(".tif", "_predseg.png")
    elif filename.endswith("tiff"):  # in filename:
        segfile = filename.replace(".tiff", "_predseg.png")
    else:
        segfile = filename
    return segfile

def prepare_output_directory(sample_direc, out_dir_name):
    out_dir_path = os.path.normpath(sample_direc + os.sep + out_dir_name)
    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)
    return out_dir_path

def prepare_segfile_name(filename, sample_direc, out_dir_path):
    segfile = get_segfile_name(filename)
    segfile = os.path.normpath(segfile)
    segfile = segfile.replace(
        os.path.normpath(sample_direc), os.path.normpath(out_dir_path)
    )
    return segfile

def do_seg(
    filename, M, metadatadict, MODEL, sample_direc, 
    NCLASSES, N_DATA_BANDS, TARGET_SIZE, TESTTIMEAUG, WRITE_MODELMETADATA,
    OTSU_THRESHOLD,
    out_dir_name='out',
    profile='minimal'
):
    """
    Perform image segmentation using the specified model.

    Args:
        filename (str): The path to the input image file.
        M (int): The number of segmentation classes.
        metadatadict (dict): A dictionary to store metadata information.
        MODEL: The segmentation model.
        sample_direc (str): The directory to hold the outputs of the models.
        NCLASSES (int): The number of classes for segmentation.
        N_DATA_BANDS (int): The number of data bands.
        TARGET_SIZE (int): The target size for image resizing.
        TESTTIMEAUG: The test time augmentation.
        WRITE_MODELMETADATA (bool): Whether to write model metadata.
        OTSU_THRESHOLD: The threshold for applying Otsu's method.
        out_dir_name (str, optional): The name of the output directory. Defaults to 'out'.
        profile (str, optional): The profiling mode. Defaults to 'minimal'.

    Returns:
        None
    """
    
    # set the name of the output file for the segmentation jpg will be saved
    segfile = get_segfile_name(filename)
    
    print("Processing image: ", filename)
    print("Output file: ", segfile)

    if WRITE_MODELMETADATA:
        metadatadict["input_file"] = filename
        
    # directory to hold the outputs of the models is named 'out' by default
    # create a directory to hold the outputs of the models, by default name it 'out' or the model name if it exists in metadatadict
    out_dir_path = prepare_output_directory(sample_direc, out_dir_name)
    # set the full location of the output file for the segmentation jpg will be saved
    segfile = prepare_segfile_name(filename, sample_direc, out_dir_path)
    
    print("Output file: ", segfile)
    print(f"NC: {NCLASSES}, NDB: {N_DATA_BANDS}, TS: {TARGET_SIZE}, TTA: {TESTTIMEAUG}, OT: {OTSU_THRESHOLD}")

    if WRITE_MODELMETADATA:
        metadatadict["nclasses"] = NCLASSES
        metadatadict["n_data_bands"] = N_DATA_BANDS

    if NCLASSES == 2:
        
        image, w, h, bigimage = process_image(filename, N_DATA_BANDS, TARGET_SIZE, MODEL)
        if image is None:
            est_label, softmax_scores = np.zeros((w, h)), np.zeros((w, h))
        else:
            est_label, softmax_scores = calculate_est_label_binary(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h)
        est_label = apply_threshold(est_label, OTSU_THRESHOLD)
    else: #NCLASSES>2
        image, w, h, bigimage = process_image(filename, N_DATA_BANDS, TARGET_SIZE, MODEL)
        if image is None:
            est_label = np.zeros((w, h))
        else:
            est_label, softmax_scores = calculate_est_label_multiclass(image, M, MODEL, TESTTIMEAUG, NCLASSES, TARGET_SIZE, w, h)
        est_label = np.argmax(softmax_scores, -1) if np.std(image) > 0 else est_label.astype('uint8')

    class_label_colormap = CLASS_LABEL_COLORMAPS[:NCLASSES]

    if WRITE_MODELMETADATA:
        metadatadict["color_segmentation_output"] = segfile

    color_label = generate_color_label(est_label, bigimage, class_label_colormap)
    save_segmentation_results(segfile, color_label)
    
    if WRITE_MODELMETADATA:
        metadatadict["grey_label"] = est_label
        print(f"writing metadata to {segfile.replace('_predseg.png', '_res.npz')}")
        np.savez_compressed(segfile.replace('_predseg.png', '_res.npz'), **metadatadict)

    if profile == 'full':
        plot_overlay(segfile.replace("_res.npz", "_overlay.png"), bigimage, color_label, N_DATA_BANDS)
        plot_side_by_side_overlay(segfile.replace("_res.npz", "_image_overlay.png"), bigimage, color_label, N_DATA_BANDS)
        plot_per_class_probabilities(segfile.replace("_res.npz", "_overlay.png"), bigimage, softmax_scores, N_DATA_BANDS)


