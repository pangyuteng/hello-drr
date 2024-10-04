import sys
import pandas as pd
import SimpleITK as sitk

csv_file = sys.argv[1]
series_instance_uid = sys.argv[2]
nifti_file = sys.argv[3]

df = pd.read_csv(csv_file)
file_list = df[df.SeriesInstanceUID==series_instance_uid].FilePath.tolist()[::-1]

reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_list)
img_obj = reader.Execute()
sitk.WriteImage(img_obj,nifti_file)


"""
/cvibraid/cvib2/apps/personal/pteng/github/hello-drr

docker run -it -u $(id -u):$(id -g) -v $PWD:/workdir -w /workdir pangyuteng/drr:latest bash

mkdir -p tmp/patient-56-files
python3 dcm2nifti.py \
    tmp/patient-56.csv \
    1.3.12.2.1107.5.1.4.60382.30000022121415511552600003945 \
    tmp/patient-56-files/ct-image.nii.gz


"""