import os
import sys
import numpy as np
import pandas as pd
import imageio
import SimpleITK as sitk
import uuid

import matplotlib.pyplot as plt
import torch

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.visualization import plot_drr, plot_mask

def main(image_nifti_file,mask_nifti_file,output_folder,device_id):

    mask_obj = sitk.ReadImage(mask_nifti_file)
    mask = sitk.GetArrayFromImage(mask_obj)
    png_file = os.path.join(output_folder,'drr-plots.png')

    subject = read(image_nifti_file,mask_nifti_file)
    print(subject.shape)

    # Initialize the DRR module for generating synthetic X-rays
    device = torch.device(device_id)
    spacing_mm = 0.4
    drr = DRR(
        subject,     # An object storing the CT volume, origin, and voxel spacing
        sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
        height=1024,  # Image height (if width is not provided, the generated DRR is square)
        width=1024,
        delx=spacing_mm,    # Pixel spacing (in mm)
        renderer='siddon', # 'siddon' or 'trilinear'
    ).to(device)

    # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

    # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
    # For example, quaternions, rotation matrix, axis-angle, etc...
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY",
        mask_to_channels=True)

    # https://github.com/eigenvivek/DiffDRR/blob/5430fa67d5599e1071806a97bed17d38c633e092/notebooks/tutorials/introduction.ipynb

    img = img.detach().cpu().numpy()
    img = np.moveaxis(img.squeeze(),0,-1)

    # save image as nifti.
    idx = 0
    drr_img = img[:,:,idx]
    drr_image_obj = sitk.GetImageFromArray(drr_img.astype(np.int32))
    drr_image_obj.SetSpacing((spacing_mm,spacing_mm))
    nifti_file = os.path.join(output_folder,'drr-image.nii.gz')
    sitk.WriteImage(drr_image_obj,nifti_file)

    # save as png
    min_val, max_val = np.min(drr_img),np.max(drr_img)
    drr_img = 255*((drr_img-min_val)/(max_val-min_val)).clip(0,1)
    drr_img_uint8 = drr_img.astype(np.uint8)
    png_file = nifti_file.replace(".nii.gz",".png")
    imageio.imwrite(png_file, drr_img_uint8)

    # save masks as nifti and also binary nifti
    drr_combined_mask = np.zeros_like(drr_img).squeeze()
    for mask_idx in sorted(list(np.unique(mask))):
        if mask_idx == 0:
            continue
        drr_slice = img[:,:,mask_idx].squeeze()
        drr_slice_obj = sitk.GetImageFromArray(drr_slice.astype(np.int32))
        drr_slice_obj.SetSpacing((spacing_mm,spacing_mm))
        nifti_file = os.path.join(output_folder,f'drr-mask-{mask_idx}.nii.gz')
        sitk.WriteImage(drr_slice_obj,nifti_file)

        drr_combined_mask[slice_mask>0]=mask_idx

    drr_mask_obj = sitk.GetImageFromArray(drr_combined_mask.astype(np.int32))
    drr_mask_obj.SetSpacing((spacing_mm,spacing_mm))
    nifti_file = os.path.join(output_folder,f'drr-mask.nii.gz')
    sitk.WriteImage(drr_mask_obj,nifti_file)

    # png_file = os.path.join(png_folder,'drr-plot.png')
    # drr_image = sitk.GetArrayFromImage(drr_image_obj)
    # drr_mask = sitk.GetArrayFromImage(drr_mask_obj)
    # plt.figure()
    # for item_png_file,idx in mydict.items():
        
    #     plt.subplot(141+idx)
    #     if idx == 0:
    #         plt.imshow(slice_img,cmap='gray')
    #         plt.colorbar()
    #     else:
    #         plt.imshow(slice_img,cmap='hot')
    #         plt.colorbar()

    # plt.savefig(png_file)

if __name__ == "__main__":
    image_nifti_file = sys.argv[1]
    mask_nifti_file = sys.argv[2]
    png_file = sys.argv[3]
    device_id = sys.argv[4] # `cpu` or `gpu` or `cuda:7`
    main(image_nifti_file,mask_nifti_file,png_file,device_id)

"""
docker run -it -v $PWD:/workdir pangyuteng/drr:latest bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

python3 gen_drr_with_mask.py \
    tmp/patient-56-files/ct-image.nii.gz \
    tmp/patient-56-files/v1_0.nii.gz \
    tmp/patient-56-files cpu

"""