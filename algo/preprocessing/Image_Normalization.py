import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def save_array_to_nifti1(array, original_img, destination_path, output_name):
    # Transform the array to a nifti image which requires the affine of the original image.
    if isinstance(original_img, nib.Nifti1Image) :
        processed_img = nib.Nifti1Image(array, original_img.affine)
    else:
        processed_img = nib.Nifti1Image(array, nib.load(original_img).affine)
    
    nib.save(processed_img, os.path.join(destination_path, output_name))

def apply_processing_to_img_folder (processing_function, source_path, destination_path, modification_string, inclusion_string="", **kwargs):
    files = os.listdir(source_path)
    
    # Select files to process.
    nifti_files = [file for file in files if (file.endswith('.nii.gz')) & (inclusion_string in file)]

    for file in nifti_files:
        file_path = os.path.join(source_path, file)
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        new_img_name = os.path.splitext(os.path.splitext(file)[0])[0] + "_" + modification_string + ".nii.gz"
    
        # Apply processing_function to the array, then save it as a nifti file.
        save_array_to_nifti1(processing_function(nii_data, **kwargs), nii_img, destination_path, new_img_name)
        print("Processed image ", file)

def z_score_normalize_saturate_outliers(array, quantile_of_max=1.0):
    # Z_score normalizing with error handling and binning of outliers at the max of the
    # specified quantile.
    saturation_value = np.quantile(array, quantile_of_max)
    array[array >= saturation_value] = saturation_value
    
    if np.std(array) != 0.0:
        return ((array-np.mean(array))/np.std(array)).astype("float32")
    else:
        return array.astype("float32")