import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt


dcm_file = sys.argv[1]
png_file = sys.argv[2]
img_obj = sitk.ReadImage(dcmfile)

img = sitk.GetArrayFromImage(img_obj)
print(img.shape)

plot_drr(img, ticks=False)
plt.show()
plt.savefig(png_file)

"""
docker run -it -v $PWD:/workdir pangyuteng/drr:latest bash

"""