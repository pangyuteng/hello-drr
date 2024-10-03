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

def main(image_nifti_file,mask_nifti_file,png_file,device_id):
    subject = read(image_nifti_file,mask_nifti_file)
    print(subject.shape)

    # Initialize the DRR module for generating synthetic X-rays
    device = torch.device(device_id)

    drr = DRR(
        subject,     # An object storing the CT volume, origin, and voxel spacing
        sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
        height=1024,  # Image height (if width is not provided, the generated DRR is square)
        width=1024,
        delx=0.4,    # Pixel spacing (in mm)
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
    mydict = {'tmp/image.png':0,'tmp/mask1.png':1,'tmp/mask2.png':2}
    plt.figure()
    for png_file,idx in mydict.items():
        slice_img = img[:,:,idx]
        min_val, max_val = np.min(slice_img),np.max(slice_img)
        slice_img = 255*((slice_img-min_val)/(max_val-min_val)).clip(0,1)
        slice_img = slice_img.astype(np.uint8)
        imageio.imwrite(png_file, slice_img)
        plt.subplot(131+idx)
        if idx == 0:
            plt.imshow(slice_img,cmap='gray')
            plt.colorbar()
        else:
            plt.imshow(slice_img,cmap='hot')
            plt.colorbar()

    plt.savefig(png_file)

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
    tmp/p19-1.3.12.2.1107.5.1.4.73191.30000018092214593450000003275-image.nii.gz \
    tmp/p19-1.3.12.2.1107.5.1.4.73191.30000018092214593450000003275-mask.nii.gz \
    tmp/test.png cpu

'tmp/p19-1.3.12.2.1107.5.1.4.73191.30000018092214593450000003275-image.nii.gz','tmp/p19-1.3.12.2.1107.5.1.4.73191.30000018092214593450000003275-mask.nii.gz','tmp/test.png','cpu'
"""