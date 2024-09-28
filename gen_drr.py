import os
import sys
import numpy as np
import pandas as pd 
import SimpleITK as sitk
import uuid

import matplotlib.pyplot as plt
import torch

from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.visualization import plot_drr

csv_file = sys.argv[1]
series_instance_uid = sys.argv[2]
png_file = sys.argv[3]
device_id = sys.argv[4] # `cpu` or `gpu` or `cuda:7`

df = pd.read_csv(csv_file)
file_list = df[df.SeriesInstanceUID==series_instance_uid].FilePath.tolist()
file_list = file_list[::-1]
reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_list)
img_obj = reader.Execute()
print(img_obj.GetSize())

nifti_file = os.path.join(f'/tmp/{uuid.uuid4().hex}.nii.gz')
sitk.WriteImage(img_obj,nifti_file)
print(nifti_file)
subject = read(nifti_file)
print(subject.shape)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device(device_id)

drr = DRR(
    subject,     # An object storing the CT volume, origin, and voxel spacing
    sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
    height=1024,  # Image height (if width is not provided, the generated DRR is square)
    delx=0.4,    # Pixel spacing (in mm)
).to(device)

# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
print(img.shape)
plot_drr(img, ticks=False)
plt.title("DRR")
plt.tight_layout()
plt.show()
plt.savefig(png_file)

"""
docker run -it -v $PWD:/workdir pangyuteng/drr:latest bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
"""