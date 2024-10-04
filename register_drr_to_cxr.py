import os
import sys
import ast
import numpy as np
import pandas as pd
import imageio
import SimpleITK as sitk
import uuid

def elastix_register_and_transform(
        fixed_image_file,fixed_spacing,
        moving_image_file,moving_spacing,moving_list=[]):

    fixed_obj = sitk.ReadImage(fixed_image_file, sitk.sitkFloat32)
    fixed_obj.SetSpacing(fixed_spacing) # or create dcm
    
    moving_obj = sitk.ReadImage(moving_image_file, sitk.sitkFloat32)
    moving_obj.SetSpacing(moving_spacing)


    fixed = sitk.GetArrayFromImage(fixed_obj)
    print('fixed',np.min(fixed),np.max(fixed))
    moving = sitk.GetArrayFromImage(moving_obj)
    print('moving',np.min(moving),np.max(moving))

    print(fixed_obj.GetSize())
    print(fixed_obj.GetSpacing())
    print(fixed_obj.GetOrigin())
    print(fixed_obj.GetDirection())
    print(moving_obj.GetSize())
    print(moving_obj.GetSpacing())
    print(moving_obj.GetOrigin())
    print(moving_obj.GetDirection())
    sys.exit(1)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)
    elastixImageFilter.SetOutputDirectory('/tmp')

    method = 'okay'
    if method == 'nonrigid':
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('nonrigid'))
    elif method == 'simple':
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
        elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
    elif method == 'okay':
        translationParameterMap = sitk.GetDefaultParameterMap("translation")
        translationParameterMap['DefaultPixelValue'] = ['0']
        translationParameterMap['Registration'] = ['MultiResolutionRegistration']
        translationParameterMap['NumberOfResolutions'] = ['5']
        translationParameterMap['FixedImagePyramidSchedule'] = ['56,56,16,16,8,8,2,2,1,1']
        translationParameterMap['MaximumNumberOfIterations'] = ['512'] 
        translationParameterMap['AutomaticTransformInitialization']=['true']
        translationParameterMap['AutomaticTransformInitializationMethod ']=['GeometricCenter']
        
        affineParameterMap = sitk.GetDefaultParameterMap("affine")
        affineParameterMap['DefaultPixelValue'] = ['0']
        affineParameterMap['MaximumNumberOfIterations'] = ['512'] 

        print(translationParameterMap)
        for k,v in  translationParameterMap.iteritems():
            print(k,v)
        print("------------")
        print(affineParameterMap)
        for k,v in  affineParameterMap.iteritems():
            print(k,v)

        print("------------")

        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(translationParameterMap)
        parameterMapVector.append(affineParameterMap)
        elastixImageFilter.SetParameterMap(parameterMapVector)
    else:
        raise NotImplementedError()

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
        og_obj = sitk.ReadImage(moving_file,sitk.sitkFloat32)
        og_obj.SetSpacing(moving_spacing)
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(og_obj)
        transformixImageFilter.SetTransformParameterMap(transform_tuple)
        transformixImageFilter.SetOutputDirectory("/tmp")
        transformixImageFilter.LogToConsoleOn()
        elastixImageFilter.LogToFileOn()
        transformixImageFilter.Execute()
        moved = transformixImageFilter.GetResultImage()
        #moved = sitk.Cast(moved,og_obj.GetPixelID())
        moved = sitk.Cast(moved, sitk.sitkUInt8)
        sitk.WriteImage(moved,moved_file)


def main(fixed_image_file,fixed_spacing,
    moving_image_file,moving_spacing,drr_mask_file,output_folder):

    moving_list = [
        dict(
            moving_file=moving_image_file,
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
    elastix_register_and_transform(
        fixed_image_file,fixed_spacing,
        moving_image_file,moving_spacing,
        moving_list=moving_list)

if __name__ == "__main__":
    cxr_image_file = sys.argv[1]
    fixed_spacing = ast.literal_eval(sys.argv[2])
    drr_image_file = sys.argv[3]
    moving_spacing = ast.literal_eval(sys.argv[4])
    drr_mask_file = sys.argv[5]
    output_folder = sys.argv[6]
    main(
        cxr_image_file,fixed_spacing,
        drr_image_file,moving_spacing,
        drr_mask_file,
        output_folder
    )

"""
docker run -it -v $PWD:/workdir \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    pangyuteng/synthmorph-wrapper:0.1.0 bash

python3 register_drr_to_cxr.py \
    tmp/patient-56-files/cxr.dcm \
    tmp/patient-56-files/drr-image.nii.gz \
    tmp/patient-56-files/drr-mask1.nii.gz \
    tmp/patient-56-files

"""