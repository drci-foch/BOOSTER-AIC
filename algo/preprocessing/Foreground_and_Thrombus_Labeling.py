import os
import glob
import nibabel as nib
import numpy as np
import scipy

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

        # Update kwargs dictionary to include file and label_directory
        kwargs["sample_filename"] = file
        
        # Apply processing_function to the array, then save it as a nifti file.
        save_array_to_nifti1(processing_function(nii_data, **kwargs), nii_img, destination_path, new_img_name)
        print("Processed image ", file)

def label_foreground_voxels (array, foreground_label=1.0, background_label=0.0, background_intensity="mode"):
    # Labels all voxels with different intensity as background.
    
    # If the default "mode" is selected as the background intensity, we use the most frequent voxel value as the background intensity value.
    # Otherwise, the value of the background must be supplied.
    if background_intensity == "mode":
        background_value = scipy.stats.mode(array, axis=None).mode
    # Otherwise the background value must be supplied
    else:
        background_value = background_intensity

    return np.where(array==background_value, background_label, foreground_label)

def label_thrombus_and_foreground_voxels (array, sample_filename, label_directory, thrombus_label=2.0, **kwargs):
    # Perform foreground labeling, then fetches thrombus label and labels thrombus on the foreground
    labeled_array = label_foreground_voxels(array, **kwargs)

    sample_number = "_".join(sample_filename.split("_")[:2])
    label_filename = glob.glob(os.path.join(label_directory, sample_number + "*"))[0]
    label_mask = nib.load(label_filename).get_fdata().astype("bool")

    labeled_array[label_mask] = thrombus_label
    
    return labeled_array.astype("float32")