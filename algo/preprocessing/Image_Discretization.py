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


def uniform_discretization (array, num_bins, quantile_of_max=1.0):
    # Set the max intensity value of the discretization. This approach sets a given quantile as the maximum, e.g. if the
    # maximum is above the quantile (say 99.9%) it is set to that value. This avoids intensities with high values but
    # very low prevalence in the data dragging the max to the right.
    max_pxl_value = np.quantile(array, quantile_of_max)

    # Set the min intensity value. Usually 0, but can be negative for z-normalized data.
    min_pxl_value = np.min(array)
    
    # Bin edges of the discretized image. The value of the pixels will be set at the middle of the bins.
    bin_edges = np.linspace(min_pxl_value, max_pxl_value, num_bins)
    bin_step = bin_edges[1] - bin_edges[0]

    # Compute the bin membership of each pixel using digitize, map it to the bin middle and shift the value respecting the
    # minimum pixel value.
    quantized_array = np.digitize(array, bin_edges) * bin_step - (0.5 * bin_step) + min_pxl_value

    # Make sure returned type is the same as input.
    return quantized_array.astype("float32")