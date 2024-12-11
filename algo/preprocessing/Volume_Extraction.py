import os
import nibabel as nib
import numpy as np
import pandas as pd

export_dir = "E:\\Data_Booster\\data_ETIS_781\\Volume_Calculation"
mask_dir = "E:\\Data_Booster\\data_ETIS_781\\MASK"
swi_dir = "E:\\Data_Booster\\data_ETIS_781\\SWI"
volume_filename = "MaskVolume_781.csv"
filename =[]
voxel_dimensions = []
volume = []

for mask, swi in zip(os.listdir(mask_dir), os.listdir(swi_dir)):
    mask_image = nib.load(os.path.join(mask_dir, mask))
    swi_image = nib.load(os.path.join(swi_dir, swi))
    mask_data = mask_image.get_fdata()
    mask_voxel_dimensions = mask_image.header["pixdim"][1:4]
    swi_voxel_dimensions = swi_image.header["pixdim"][1:4]

    if np.sum(np.abs(mask_voxel_dimensions - swi_voxel_dimensions)) < 0.1:
        filename.append("_".join(mask.split("_")[:2]))
        voxel_dimensions.append(mask_voxel_dimensions)
        volume.append(np.product(mask_voxel_dimensions) * np.sum(mask_data))
    else:
        print(f"Mismatch in voxel dimensions for file {mask}")

volume_dataframe = pd.DataFrame(data={"Voxel Dimensions [X,Y,Z] (mm)":voxel_dimensions, "Volume (mm^3)":volume}, index=pd.Series(filename, name="Filename"))
volume_dataframe.to_csv(os.path.join(export_dir, volume_filename))