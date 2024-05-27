import nibabel as nib
import numpy as np
import torchio as tio
import os

def tuple_product(*args):
    product = 1
    for element in args:
        product *= element
    return product

def resize_images_to_reference (source_path, destination_path, ref_image_path=None, modification_string="", inclusion_string="", save_ref=False):
    files = os.listdir(source_path)

    # Select files to process.
    nifti_files = [file for file in files if (file.endswith('.nii.gz')) & (inclusion_string in file)]
    nii_img_shapes = []

    # First get a list of image shapes.
    for file in nifti_files:
        file_path = os.path.join(source_path, file)
        nii_img = nib.load(file_path)
        nii_img_shapes.append(nii_img.shape)
    
    nifti_files_to_process = nifti_files.copy()

    # If no reference image is used, dynamically choose a reference among selected files by choosing the biggest image.
    if ref_image_path is None:
        index_of_max_product = max(range(len(nii_img_shapes)), key=lambda i: tuple_product(nii_img_shapes[i]))
        del nifti_files_to_process[index_of_max_product]
        
        reference_filename = nifti_files[index_of_max_product]
        reference_image = tio.ScalarImage(os.path.join(source_path, reference_filename))
        
    else:
        reference_filename = os.path.basename(ref_image_path)
        reference_image = tio.ScalarImage(ref_image_path)
    
    # Define the reference image name, adding the modification string if necessary.
    split_name = reference_filename.split("_")
    if modification_string != "":
        split_name.insert(-1, modification_string)
    new_reference_img_name = "_".join(split_name)

    if save_ref:
        reference_image.save(os.path.join(destination_path, new_reference_img_name))

    print("Processed reference image ", reference_filename)

    # Use tio.Resample to resize all images to match the shape of the reference image.
    for file in nifti_files_to_process:
        file_path = os.path.join(source_path, file)
        
        image_to_resize = tio.ScalarImage(file_path)
        normalized_image = tio.Resample(target=reference_image)(image_to_resize)
        
        split_name = file.split("_")
        if modification_string != "":
            split_name.insert(-1, modification_string)
        new_img_name = "_".join(split_name)

        normalized_image.save(os.path.join(destination_path, new_img_name))
        
        print("Processed image ", file)