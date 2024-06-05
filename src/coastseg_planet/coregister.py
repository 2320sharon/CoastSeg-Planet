from arosics import COREG, DESHIFTER,COREG_LOCAL
import numpy as np
import geowombat as gw
import glob
import os
import sys

def get_cpus():
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    return num_cpus

def global_coregister_gw(TARGET, REFERENCE, OUTPUT_PATH,target_cloud_mask:str, num_cpus=1, kwargs={}):
    """
    Coregisters the target image with the reference image using the Geowave library.

    Args:
        TARGET (str): Path to the target image file.
        REFERENCE (str): Path to the reference image file.
        OUTPUT_PATH (str): Path to save the coregistered image.
        num_cpus (int, optional): Number of CPUs to use during pixel grid equalization. Defaults to 1.
        kwargs (dict, optional): Additional keyword arguments for the coregistration process. Defaults to {}.

    Returns:
        None
    """
    # This is global coregistration since within the geowombat code it shows it calling COREG and not COREG_LOCAL
    with gw.open(TARGET) as target, gw.open(REFERENCE) as reference:
        target_shifted = gw.coregister(
            target=target,
            reference=reference,
            max_shift=500,
            q=False,  # quiet mode (default: False)
            CPUs=num_cpus,  # number of CPUs to use during pixel grid equalization (default: None, which means ‘all CPUs available’)
            mask_baddata_tgt = target_cloud_mask,
        )
        print(target_shifted)
        gw.save(target_shifted, OUTPUT_PATH, compress='lzw', num_workers=8)
    return OUTPUT_PATH

def coregister_arosics_global(target_path:str, reference_path:str, output_path:str, target_cloud_mask:str,quiet_mode=False, max_shift=200, align_grids=True,num_cpus=1):
    "documentation at https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#module-arosics.CoReg"
    CR = COREG( reference_path,
            target_path,
            path_out=output_path,
            mask_baddata_tgt = target_cloud_mask,
            max_shift=max_shift, 
            align_grids = align_grids, # align the input coordinate grid to the reference (does not affect the output pixel size as long as input and output pixel sizes are compatible (5:30 or 10:30 but not 4:30))
            fmt_out='GTIFF',
            out_crea_options=['COMPRESS=DEFLATE'],
            q=quiet_mode,
            CPUs=num_cpus,)
    print(f"ssim_improved : {CR.ssim_improved}")
    CR.correct_shifts()

def coregister_arosics_local(target_path:str, reference_path:str,reference_cloud_mask:str, output_path:str, target_cloud_mask:str,quiet_mode=False,num_cpus:int=1,**kwargs):
    "documentation at https://danschef.git-pages.gfz-potsdam.de/arosics/doc/arosics.html#module-arosics.CoReg_local"

    defaults = {
    'max_shift':30,
    'align_grids':True,
    'grid_res':5, # Grid points are 5 meters apart
    'v':True, #verbose mode
    'q':False, #quiet mode
    'ignore_errors':False, # Recommended to set to True for batch processing
    'progress':True,
    'out_crea_options':["COMPRESS=LZW"], # otherwise if will use deflate which will not work well with our model
    'fmt_out':'GTIFF',
    'CPUs' : get_cpus(),
    }
    
    
    
    # Update the defaults with the user-provided kwargs
    defaults.update(kwargs)
    print(defaults)
    CR = COREG_LOCAL(  reference_path,
            target_path,
            path_out=output_path,
            mask_baddata_tgt = target_cloud_mask,
            mask_baddata_ref = reference_cloud_mask,
            **defaults)
    CR.correct_shifts()