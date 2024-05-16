from arosics import COREG, DESHIFTER
import numpy as np
import geowombat as gw

import glob
import os
import sys

def get_cpus():
    num_cpus = os.cpu_count()
    print(f"Number of CPUs available: {num_cpus}")
    return num_cpus

def local_coregister_gw(TARGET, REFERENCE, OUTPUT_PATH,target_cloud_mask:str, num_cpus=1, kwargs={}):
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
    # I'm actually not sure if this is local or global coregistration
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