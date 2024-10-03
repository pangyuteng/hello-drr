import os
import sys
import numpy as np
import pandas as pd
import imageio
import SimpleITK as sitk
import uuid

def elastix_register_and_transform(fixed_image_file,moving_image_file,moving_list=[]):

    fixed = sitk.ReadImage(fixed_image_file)
    moving = sitk.ReadImage(moving_image_file)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetOutputDirectory('/tmp')
    
    defaultTranslationParameterMap = sitk.GetDefaultParameterMap("translation")
    # TODO: hard code bad
    defaultTranslationParameterMap['DefaultPixelValue'] = ['0']
    defaultTranslationParameterMap['MaximumNumberOfIterations'] = ['512'] 
    defaultAffineParameterMap = sitk.GetDefaultParameterMap("affine")
    defaultAffineParameterMap['DefaultPixelValue'] = ['0']
    defaultAffineParameterMap['MaximumNumberOfIterations'] = ['512'] 
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(defaultTranslationParameterMap)
    parameterMapVector.append(defaultAffineParameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    for item in moving_list:
        moving_file = item["moving_file"]
        moved_file = item["moved_file"]
        out_pixel_value = item["out_val"]
        is_mask = item["is_mask"]

        transform_tuple = elastixImageFilter.GetTransformParameterMap()
        transform = list(transform_tuple)
        transform[-1]['DefaultPixelValue']=[str(out_pixel_value)]
        if is_mask:
            transform[-1]['FinalBSplineInterpolationOrder']=["0"]
            transform[-1]["ResultImagePixelType"] = ["int"]    
        # 
        # TODO: maybe something funky here? with int transformration
        # 
        og_obj = sitk.ReadImage(moving_file)
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(og_obj)
        transformixImageFilter.SetTransformParameterMap(transform_tuple)
        transformixImageFilter.SetOutputDirectory("/tmp")
        transformixImageFilter.LogToConsoleOn()
        elastixImageFilter.LogToFileOn()
        transformixImageFilter.Execute()
        moved = transformixImageFilter.GetResultImage()
        moved = sitk.Cast(moved,og_obj.GetPixelID())
        sitk.WriteImage(moved,moved_file)


def main(cxr_image_file,drr_image_file,drr_mask_file,output_folder):

    moving_list = [
        dict(
            moving_file=drr_image_file,
            moved_file=os.path.join(output_folder,'moved-drr-image.png'),
            out_val=0,
            is_mask=False
        ),
        dict(
            moving_file=drr_mask_file,
            moved_file=os.path.join(output_folder,'moved-drr-mask-test.png'),
            out_val=0,
            is_mask=True
        ),
    ]
    elastix_register_and_transform(cxr_image_file,drr_image_file,moving_list=moving_list)

if __name__ == "__main__":
    cxr_image_file = sys.argv[1]
    drr_image_file = sys.argv[2]
    drr_mask_file = sys.argv[3]
    output_folder = sys.argv[4]
    main(cxr_image_file,drr_image_file,drr_mask_file,output_folder)

"""
docker run -it -v $PWD:/workdir \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    pangyuteng/synthmorph-wrapper:0.1.0 bash

python3 register_drr_to_cxr.py \
    tmp/patient-56-files/cxr.png \
    tmp/patient-56-files/drr-image.png \
    tmp/patient-56-files/drr-mask1-binary.png \
    tmp/patient-56-files


"""