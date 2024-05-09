# I should be able to co-register these images now
from arosics import COREG, DESHIFTER
import glob
import os
import sys

REFERENCE= r"CoastSeg-Planet\ID_mqq51_sample_int3.tif" 
TARGET = r"CoastSeg-Planet\2023-02-14-10-33-43_L8_ID_mqq51_unit16.tif"
outfile= r"CoastSeg-Planet\coregistered_landsat_2023-02-14-10-33-43_L8_ID_mqq51_unit16.tif"

print(os.path.exists(REFERENCE))
print(os.path.exists(TARGET))

# detect and correct global spatial shift 
#Geometric shifts are calculated at a specific (adjustable) image position. Correction performs a global shifting in X- or Y direction
CR = COREG(REFERENCE, TARGET,
           path_out=outfile,
           max_shift=50, 
           align_grids = True, # align the input coordinate grid to the reference (does not affect the output pixel size as long as input and output pixel sizes are compatible (5:30 or 10:30 but not 4:30))
           fmt_out='GTIFF',
           out_crea_options=['COMPRESS=DEFLATE'],
           r_b4match=1, 
           s_b4match=1)

print(CR)

#Correct the already calculated X/Y shift of the target image. Returns a dictionary of the deshift results
print(f"CR.correct_shifts(): {CR.correct_shifts()}")

# Return True if image similarity within the matching window has been improved by co-registration.
print(f"CR.ssim_improved: {CR.ssim_improved}")
