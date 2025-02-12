import nibabel as nib
import torchio as tio
import ants
import os

def tuple_product(*args):
    product = 1
    for element in args:
        product *= element
    return product

def resize_images_to_reference (source_path, destination_path, ref_image_path=None, image_is_label=False, ref_spacing=None, transform_to_canonical=False, interpolation_method="linear", modification_string="", inclusion_string="", save_ref=False):
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

    # No need to add a reference image if a spacing is given
    if ref_spacing != None:
        pass
    # If no reference image is used, dynamically choose a reference among selected files by choosing the biggest image.
    elif ref_image_path is None:
        index_of_max_product = max(range(len(nii_img_shapes)), key=lambda i: tuple_product(nii_img_shapes[i]))
        del nifti_files_to_process[index_of_max_product]
        
        reference_filename = nifti_files[index_of_max_product]
        reference_image = tio.ScalarImage(os.path.join(source_path, reference_filename))
        
    else:
        reference_filename = os.path.basename(ref_image_path)
        reference_image = tio.ScalarImage(ref_image_path)
    
    if ref_spacing is None:
        # Define the reference image name, adding the modification string if necessary.
        split_name = reference_filename.split(".")
        if modification_string != "":
            split_name[0] = split_name[0] + "_" + modification_string
        new_reference_img_name = ".".join(split_name)
    
        if save_ref:
            reference_image.save(os.path.join(destination_path, new_reference_img_name))

        print("Processed reference image ", reference_filename)

    # Use tio.Resample to resize all images to match the shape of the reference image.
    for file in nifti_files_to_process:
        file_path = os.path.join(source_path, file)

        if image_is_label:
            image_to_resize = tio.LabelMap(file_path)
        else:
            image_to_resize = tio.ScalarImage(file_path)
       # 4 cases for the combinations of ref_spacing None/user-supplied and transform_to_canonical True/False.
        if (ref_spacing is None) & (transform_to_canonical):
            transforms = tio.ToCanonical(), tio.Resample(target=reference_image)
            total_transform = tio.Compose(transforms)
            normalized_image = total_transform(image_to_resize)
        elif (ref_spacing is None) & (transform_to_canonical == False):
            normalized_image = tio.Resample(target=reference_image)(image_to_resize)
        elif (ref_spacing is not None) & (transform_to_canonical):
            transforms = tio.ToCanonical(), tio.Resample(target=ref_spacing, image_interpolation=interpolation_method)
            total_transform = tio.Compose(transforms)
            normalized_image = total_transform(image_to_resize)
        else:
            normalized_image = tio.Resample(target=ref_spacing, image_interpolation=interpolation_method)(image_to_resize)
        
        split_name = file.split(".")
        if modification_string != "":
            split_name[0] = split_name[0] + "_" + modification_string
        new_img_name = ".".join(split_name)

        normalized_image.save(os.path.join(destination_path, new_img_name))
        
        print("Processed image ", file)

def resize_images_to_moving_reference (source_path, target_path, ref_folder_path, image_is_label=False, transform_to_canonical=False, interpolation_method="linear", modification_string="", inclusion_string=""):
    # Resize a folder of images according to a reference of another folder of images. There must be a 1-to-1 correspondence between the resized and reference folder.

    # Select files to process.
    nifti_files_source = [file for file in os.listdir(source_path) if (file.endswith('.nii.gz')) & (inclusion_string in file)]
    nifti_files_ref = [file for file in os.listdir(ref_folder_path) if (file.endswith('.nii.gz')) & (inclusion_string in file)]

    if len(nifti_files_source) != len(nifti_files_ref):
        raise SystemExit("Number of images to resize and their reference is different.")
    
    for source_file, ref_file in zip(nifti_files_source, nifti_files_ref):
        source_file_path = os.path.join(source_path, source_file)
        ref_file_path = os.path.join(ref_folder_path, ref_file)

        if image_is_label:
            image_to_resize = tio.LabelMap(source_file_path)
            ref_image = tio.ScalarImage(ref_file_path)
            if transform_to_canonical:
                transforms = tio.ToCanonical(), tio.Resample(target=ref_image)
                total_transform = tio.Compose(transforms)
                normalized_image = total_transform(image_to_resize)
            else:
                normalized_image = tio.Resample(target=ref_image)(image_to_resize)
        else:
            image_to_resize = tio.ScalarImage(source_file_path)
            ref_image = tio.ScalarImage(ref_file_path)
            if transform_to_canonical:
                transforms = tio.ToCanonical(), tio.Resample(target=ref_image, image_interpolation=interpolation_method)
                total_transform = tio.Compose(transforms)
                normalized_image = total_transform(image_to_resize)
            else:
                normalized_image = tio.Resample(target=ref_image, image_interpolation=interpolation_method)(image_to_resize)

        split_name = source_file.split(".")
        if modification_string != "":
            split_name[0] = split_name[0] + "_" + modification_string
        
        new_img_name = ".".join(split_name)
        normalized_image.save(os.path.join(target_path, new_img_name))
        print("Processed image ", source_file)

def register_mask_to_moving_reference (mask_source_path, target_path, ref_folder_path, atlas_path, registration_type="AffineFast", interpolation_method="nearestNeighbor", modification_string="", inclusion_string=""):
    # Register a folder of images according to a reference of another folder of images. Then perform the registration on a list of masks to obtain the registered mask.
    # There must be a 1-to-1 correspondence between the registered and reference folder.

    # Select files to process.
    nifti_files_source = [file for file in os.listdir(mask_source_path) if (file.endswith('.nii.gz')) & (inclusion_string in file)]
    nifti_files_ref = [file for file in os.listdir(ref_folder_path) if (file.endswith('.nii.gz')) & (inclusion_string in file)]

    if len(nifti_files_source) != len(nifti_files_ref):
        raise SystemExit("Number of images to resize and their reference is different.")
    
    for source_file, ref_file in zip(nifti_files_source, nifti_files_ref):
        source_file_path = os.path.join(mask_source_path, source_file)
        ref_file_path = os.path.join(ref_folder_path, ref_file)

        atlas_image = ants.image_read(atlas_path)
        atlas_mask = ants.image_read(source_file_path)
        target_image = ants.image_read(ref_file_path)
        registration_results = ants.registration(fixed=target_image, moving=atlas_image, type_of_transform=registration_type)
        transformed_mask = ants.apply_transforms(fixed=target_image, moving=atlas_mask, transformlist=registration_results["fwdtransforms"], interpolator=interpolation_method)

        split_name = source_file.split(".")
        if modification_string != "":
            split_name[0] = split_name[0] + "_" + modification_string
        
        new_img_name = ".".join(split_name)
        ants.image_write(transformed_mask, os.path.join(target_path, new_img_name))
        print("Processed image ", source_file)

def transform_images_to_canonical (source_path, target_path, image_is_label=False, modification_string="", inclusion_string=""):
    # Transform a folder of images to canonical frame of reference as defined by torchio ToCanonical.

    # Select files to process.
    nifti_files_source = [file for file in os.listdir(source_path) if (file.endswith('.nii.gz')) & (inclusion_string in file)]
    for source_file in nifti_files_source:
        source_file_path = os.path.join(source_path, source_file)

        if image_is_label:
            image_to_resize = tio.LabelMap(source_file_path)
            normalized_image = tio.ToCanonical()(image_to_resize)
        else:
            image_to_resize = tio.ScalarImage(source_file_path)
            normalized_image = tio.ToCanonical()(image_to_resize)
        
        split_name = source_file.split(".")
        if modification_string != "":
            split_name[0] = split_name[0] + "_" + modification_string
        
        new_img_name = ".".join(split_name)
        normalized_image.save(os.path.join(target_path, new_img_name))
        print("Processed image ", source_file)