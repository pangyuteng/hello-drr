import sys
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from diffdrr.visualization import plot_drr
import imageio

#csv_file = sys.argv[1]
#series_instance_uid = sys.argv[2]
#png_file = sys.argv[3]
#df = pd.read_csv(csv_file)
#dcm_file = df[df.SeriesInstanceUID==series_instance_uid].iloc[-1].FilePath
# img_obj = sitk.ReadImage(dcm_file)
# img = sitk.GetArrayFromImage(img_obj)
# img = np.squeeze(img)


dcm_file = sys.argv[1]
png_file = sys.argv[2]
print(dcm_file)
ds = pydicom.dcmread(dcm_file)
img = ds.pixel_array
print(img.shape)

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title("CXR")
plt.tight_layout()
plt.show()
plt.savefig(png_file)

hr_png_file = png_file+".hr.png"

min_val, max_val = np.min(img),np.max(img)
img = 255*((img-min_val)/(max_val-min_val)).clip(0,1)
img = img.astype(np.uint8)
imageio.imwrite(hr_png_file, img)

"""
docker run -it -u $(id -u):$(id -g) -v $PWD:/workdir -w /workdir pangyuteng/drr:latest bash

"""