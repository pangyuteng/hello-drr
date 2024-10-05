
import os
import sys
import imageio
import numpy as np
import pandas as pd
import SimpleITK as sitk
import uuid
import matplotlib.pyplot as plt


# save as png
min_val, max_val = np.min(drr_img),np.max(drr_img)
drr_img = 255*((drr_img-min_val)/(max_val-min_val)).clip(0,1)
drr_img_uint8 = drr_img.astype(np.uint8)
png_file = nifti_file.replace(".nii.gz",".png")
imageio.imwrite(png_file, drr_img_uint8)
