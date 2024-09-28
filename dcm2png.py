import sys
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from diffdrr.visualization import plot_drr


csv_file = sys.argv[1]
series_instance_uid = sys.argv[2]
png_file = sys.argv[3]
df = pd.read_csv(csv_file)
dcm_file = df[df.SeriesInstanceUID==series_instance_uid].iloc[-1].FilePath
print(dcm_file)
img_obj = sitk.ReadImage(dcm_file)

img = sitk.GetArrayFromImage(img_obj)
img = np.squeeze(img)

#plot_drr(img, ticks=False)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.title("CXR")
plt.show()
plt.savefig(png_file)

"""
docker run -it -v $PWD:/workdir pangyuteng/drr:latest bash

"""