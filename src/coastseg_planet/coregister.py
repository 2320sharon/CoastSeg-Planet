from arosics import COREG, DESHIFTER, COREG_LOCAL
import numpy as np

# import geowombat as gw
import glob
import os
from coastseg_planet.processing import get_base_filename
from coastseg_planet import utils


def get_cpus():
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    return num_cpus


def coregister_directory(
    directory: str,
    landsat_path: str,
    landsat_cloud_mask_path: str,
    input_suffix: str = "_TOAR_model_format.tif",
    output_suffix: str = "_TOAR_processed_coregistered.tif",
    separator="_3B",
    bad_mask_suffix:str = "udm2_clip_combined_mask.tif",
    use_local: bool = True,
    overwrite: bool = False,
    **kwargs,
):
    """
    Coregisters multiple planet images in a directory to Landsat images.

    Args:
        directory (str): The directory containing the planet images.
        landsat_path (str): The path to the Landsat image.
        landsat_cloud_mask_path (str): The path to the Landsat cloud mask image.
        input_suffix (str, optional): The suffix of the planet images to be coregistered. Defaults to "*_TOAR_model_format.tif".
        output_suffix (str, optional): The suffix to be added to the coregistered planet images. Defaults to "_TOAR_processed_coregistered.tif".
        separator (str, optional): The separator used in the planet image filenames. Defaults to "_3B".
        bad_mask_suffix (str, optional): The suffix of the bad mask image. Defaults to "udm2_clip_combined_mask.tif".
        use_local (bool, optional): Flag indicating whether to use local files for coregistration. Defaults to True.
        overwrite (bool, optional): Flag indicating whether to overwrite existing coregistered images. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the coregister_tiff function.

    Returns:
        None
    """
    defaults = {
        "max_shift": 2,
        "align_grids": True,
        "grid_res": 100,  # Grid points are 30 meters apart (the lower this value the longer it takes)
        "v": True,  # verbose mode
        "q": False,  # quiet mode
        "min_reliability": 50,  # minimum reliability of the tie points is 50%
        "ignore_errors": False,  # Recommended to set to True for batch processing
        "progress": True,
        "out_crea_options": [
            "COMPRESS=LZW"
        ],  # otherwise if will use deflate which will not work well with our model
        "fmt_out": "GTIFF",
        "CPUs": get_cpus() / 2,
    }
    defaults.update(kwargs)

    inputs_paths = glob.glob(os.path.join(directory, f"*{separator}{input_suffix}"))
    if len(inputs_paths) == 0:
        print(f"No files found with suffix {input_suffix} in {directory}")
        return
    if len(inputs_paths) == 1:
        input_path = inputs_paths[0]
        output_path = coregister_tiff(
            input_path,
            landsat_path,
            landsat_cloud_mask_path,
            output_suffix,
            bad_mask_suffix,
            separator,
            use_local,
            overwrite,
            **kwargs,
        )
        if output_path:
            print(f"Coregistered planet image saved to: {output_path}")
        else:
            print(f" coregister_directory len1 Error processing {input_path}")
    for input_path in inputs_paths:
        output_path = coregister_tiff(
            input_path,
            landsat_path,
            landsat_cloud_mask_path,
            output_suffix,
            bad_mask_suffix,
            separator,
            use_local,
            overwrite,
            **kwargs,
        )
        print(f"output_path: {output_path}")
        if output_path:
            print(f"Coregistered planet image saved to: {output_path}")
        else:
            print(f" coregister_directory Error processing {input_path}")


def coregister_tiff(
    input_path: str,
    landsat_path: str,
    landsat_cloud_mask_path: str,
    output_suffix: str = "_TOAR_processed_coregistered.tif",
    bad_mask_suffix:str = "udm2_clip_combined_mask.tif",
    separator="_3B",
    use_local: bool = True,
    overwrite: bool = False,
    **kwargs,
):  # Fixed the kwargs syntax
    print(f"bad_mask_suffix: {bad_mask_suffix}")
    print(f"input_path: {input_path}")
    parent_dir = os.path.dirname(input_path)
    print(f"Parent directory: {parent_dir}")
    if os.path.exists(input_path) == False:
        print(f"Could not find {input_path}")
        return
    base_filename = get_base_filename(input_path, separator)
    target_cloud_mask_path = utils.get_file_path(
        parent_dir, base_filename, regex=f"*{bad_mask_suffix}"
    )
    if not target_cloud_mask_path:
        print(f"Could not find cloud mask for {os.path.basename(input_path)}")
        return
    # make the output path
    output_path = os.path.join(parent_dir, f"{base_filename}{separator}{output_suffix}")
    # return the output path if it already exists
    if not overwrite and os.path.exists(output_path):
        # os.remove(output_path)
        return output_path
    try:
        if use_local:
            coregistered_target_path = coregister_arosics_local(
                input_path,
                landsat_path,
                landsat_cloud_mask_path,
                output_path,
                target_cloud_mask=target_cloud_mask_path,
                **kwargs,
            )
        else:
            # perform global coregistration
            coregistered_target_path = coregister_arosics_global(
                input_path,
                landsat_path,
                landsat_cloud_mask_path,
                output_path,
                target_cloud_mask=target_cloud_mask_path,
                **kwargs,
            )
    except Exception as e:
        print(f"Error processing {e} : {input_path}")
        if coregistered_target_path:
            return coregistered_target_path
        else:
            return None
    print(f"Coregistered planet image saved to: {coregistered_target_path}")


def coregister_arosics_global(
    target_path: str,
    reference_path: str,
    reference_cloud_mask: str,
    output_path: str,
    target_cloud_mask: str,
    **kwargs,
):
    """
    Perform global coregistration using AROSICS library.
    Args:
        target_path (str): Path to the target image.
        reference_path (str): Path to the reference image.
        reference_cloud_mask (str): Path to the reference cloud mask.
        output_path (str): Path to save the output image.
        target_cloud_mask (str): Path to the target cloud mask.
        **kwargs: Additional keyword arguments to customize the coregistration process.
            max_shift (float): Maximum shift allowed during coregistration. Defaults to 5.
            align_grids (bool): Whether to align the input coordinate grid to the reference. Defaults to True.
            v (bool): Verbose mode. Defaults to True.
            q (bool): Quiet mode. Defaults to False.
            ignore_errors (bool): Whether to ignore errors during batch processing. Defaults to False.
            progress (bool): Whether to show progress during coregistration. Defaults to True.
            out_crea_options (list): Output creation options. Defaults to ["COMPRESS=LZW"].
            fmt_out (str): Output format. Defaults to 'GTIFF'.
            CPUs (int): Number of CPUs to use during coregistration. Defaults to the number of available CPUs.
    Returns:
        str: Path to the output image.
    More information about the AROSICS library and its parameters can be found at:
    https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#module-arosics.CoReg
    """

    defaults = {
        "max_shift": 5,
        "align_grids": True,
        "v": True,  # verbose mode
        "q": False,  # quiet mode
        "ignore_errors": False,  # Recommended to set to True for batch processing
        "progress": True,
        "out_crea_options": [
            "COMPRESS=LZW"
        ],  # otherwise if will use deflate which will not work well with our model
        "fmt_out": "GTIFF",
        "CPUs": get_cpus(),
    }

    # Update the defaults with the user-provided kwargs
    defaults.update(kwargs)

    if "grid_res" in kwargs:
        # remove this parameter because it is not used in the global coregistration
        del defaults["grid_res"]

    if "tieP_filter_level" in kwargs:
        # remove this parameter because it is not used in the global coregistration
        del defaults["tieP_filter_level"]

    if "min_reliability" in kwargs:
        # remove this parameter because it is not used in the global coregistration
        del defaults["min_reliability"]

    if "max_points" in kwargs:
        # remove this parameter because it is not used in the global coregistration
        del defaults["max_points"]

    CR = COREG(
        reference_path,
        target_path,
        path_out=output_path,
        mask_baddata_tgt=target_cloud_mask,
        mask_baddata_ref=reference_cloud_mask,
        **defaults,
    )
    print(f"ssim_improved : {CR.ssim_improved}")
    CR.correct_shifts()
    return output_path


# def global_coregister_gw(TARGET, REFERENCE, OUTPUT_PATH,target_cloud_mask:str, num_cpus=1, kwargs={}):
#     """
#     Coregisters the target image with the reference image using the Geowave library.

#     Args:
#         TARGET (str): Path to the target image file.
#         REFERENCE (str): Path to the reference image file.
#         OUTPUT_PATH (str): Path to save the coregistered image.
#         num_cpus (int, optional): Number of CPUs to use during pixel grid equalization. Defaults to 1.
#         kwargs (dict, optional): Additional keyword arguments for the coregistration process. Defaults to {}.

#     Returns:
#         None
#     """
#     # This is global coregistration since within the geowombat code it shows it calling COREG and not COREG_LOCAL
#     with gw.open(TARGET) as target, gw.open(REFERENCE) as reference:
#         target_shifted = gw.coregister(
#             target=target,
#             reference=reference,
#             max_shift=500,
#             q=False,  # quiet mode (default: False)
#             CPUs=num_cpus,  # number of CPUs to use during pixel grid equalization (default: None, which means ‘all CPUs available’)
#             mask_baddata_tgt = target_cloud_mask,
#         )
#         print(target_shifted)
#         gw.save(target_shifted, OUTPUT_PATH, compress='lzw', num_workers=8)
#     return OUTPUT_PATH


def coregister_arosics_local(
    target_path: str,
    reference_path: str,
    reference_cloud_mask: str,
    output_path: str,
    target_cloud_mask: str,
    quiet_mode=False,
    num_cpus: int = 1,
    **kwargs,
):
    "documentation at https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#module-arosics.CoReg_local"

    defaults = {
        "max_shift": 2,
        "align_grids": True,
        "grid_res": 10,  # Grid points are 10 meters apart
        "v": True,  # verbose mode
        "q": False,  # quiet mode
        "ignore_errors": False,  # Recommended to set to True for batch processing
        "progress": True,
        "out_crea_options": [
            "COMPRESS=LZW"
        ],  # otherwise if will use deflate which will not work well with our model
        "fmt_out": "GTIFF",
        "CPUs": get_cpus(),
    }

    # Update the defaults with the user-provided kwargs
    defaults.update(kwargs)
    print(f"default: {defaults}")
    CR = COREG_LOCAL(
        reference_path,
        target_path,
        path_out=output_path,
        mask_baddata_tgt=target_cloud_mask,
        mask_baddata_ref=reference_cloud_mask,
        **defaults,
    )
    CR.correct_shifts()
    return output_path
