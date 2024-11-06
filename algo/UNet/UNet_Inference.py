import torch
import torchio as tio
import os
import sys
import nibabel as nib
import numpy as np
import monai
import tqdm

# Helper Function

def save_array_to_nifti1(array, original_img, destination_path, output_name):
    # Transform the array to a nifti image which requires the affine of the original image.
    if isinstance(original_img, nib.Nifti1Image) :
        processed_img = nib.Nifti1Image(array, original_img.affine)
    else:
        processed_img = nib.Nifti1Image(array, nib.load(original_img).affine)
    
    nib.save(processed_img, os.path.join(destination_path, output_name))

def remove_padding_from_tensor(tensor, original_dims):
    # Apply the padding removal function to the tensor to restore the original dimensions
    current_dims = tensor.shape[-len(original_dims):]
    resized_tensor = tensor
    
    for dim, (original_size, current_size) in enumerate(zip(original_dims, current_dims), start=-len(original_dims)):
        total_padding = current_size - original_size
        padding_before = total_padding // 2
        
        resized_tensor = torch.narrow(resized_tensor, dim, padding_before, original_size)
    return resized_tensor

def logit_to_binary_mask(tensor, threshold=0.5):
    # Transform a tensor of logits into a binary mask according to the specified probability threshold.
    mask_tensor = torch.sigmoid(tensor)

    return (mask_tensor >= threshold).float()

def restore_inference_original_size (prediction_list, original_img_dir):
    return [torch.squeeze(
        remove_padding_from_tensor(prediction, nib.load(os.path.join(original_img_dir, original_file)).shape)
        )
        for prediction, original_file in zip(prediction_list, os.listdir(original_img_dir))
        ]

def load_prediction_masks (mask_dir):
    return [torch.tensor(nib.load(os.path.join(mask_dir, mask)).get_fdata()) for mask in os.listdir(mask_dir) if (mask.endswith(".nii.gz"))]

def pad_collate_fn(batch):
    # Collate function to ensure batches have the same size, that is the size of the largest image in the batch.
    # Find the max shape in each dimension from the batch
    max_dims = [max([subject['swi_image'][tio.DATA].shape[-3:][d] for subject in batch]) for d in range(3)]
    
    # Apply padding to each image in the batch to match the largest dimensions
    padded_subjects = []
    for subject in batch:
        subject_image = subject['swi_image'][tio.DATA]  # Get the image tensor
        
        # Calculate how much padding is needed for each dimension
        padding = []
        for max_dim, current_size in zip(max_dims, subject_image.shape[-3:]):
            total_padding = max_dim - current_size
            # Divide padding evenly across both sides
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before  # Handle the case where the difference is odd
            padding.extend([padding_before, padding_after])
                
        # Apply padding to the image
        pad_transform = tio.Pad(padding)
        padded_image = pad_transform(subject['swi_image'])
        
        # Create a new Subject with the padded image
        padded_subject = tio.Subject(swi_image=tio.ScalarImage(tensor=padded_image[tio.DATA]))
        padded_subjects.append(padded_subject)
    
    return padded_subjects

def save_predictions (predictions_list, save_destination, filename_source_dir):
    if not os.path.exists(save_destination):
        os.makedirs(save_destination)
        print(f"Folder created at {save_destination}")
    else:
        if os.listdir(save_destination):
            print(f"Error: Save location at {save_destination} is not empty.")
            sys.exit(1)
            
    for mask, filename in zip(predictions_list, os.listdir(filename_source_dir)):
        output_mask_name = "_".join(filename.split("_")[:2]) + "_Prediction" + ".nii.gz"
        save_array_to_nifti1(np.array(mask), os.path.join(filename_source_dir, filename), save_destination, output_mask_name)

def compute_dice_metric (prediction_list, ground_truth_masks_list):
    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_score_list = np.array(
        [dice_metric(prediction.unsqueeze(0).unsqueeze(0), gt_mask.unsqueeze(0).unsqueeze(0))
         for prediction, gt_mask in zip(prediction_list, ground_truth_masks_list)]
        ).flatten()
    print(f"Mean Dice Score is {dice_score_list.mean()}")
    return dice_score_list

def compute_sensitivity_metric (prediction_list, ground_truth_masks_list):
    intersection_list = np.array(
        [((prediction * gt_mask).sum()/gt_mask.sum()) for prediction, gt_mask in zip(prediction_list, ground_truth_masks_list)]
    ).flatten()
    print(f"Mean Sensitivity is {intersection_list.mean()}")
    return intersection_list


# Dataset Classes
class HighToleranceSubject(tio.Subject):
    # A custom instance of Subject with higher tolerance in attribute checking.
    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-3
        kwargs['absolute_tolerance'] = 1e-3
        return super().check_consistent_attribute(*args, **kwargs)

def load_subjectsdataset_1channel (swi_dir, thrombus_labels_dir, foreground_labels_dir, **kwargs):
    subjects_list = []
    swi_list = os.listdir(swi_dir)
    thrombus_labels_list = os.listdir(thrombus_labels_dir)
    foreground_labels_list = os.listdir(foreground_labels_dir)
    
    if len(swi_list) != len(thrombus_labels_list) != len(foreground_labels_list):
        print("Mismatch in sample numbers")
    
    for swi_file, thrombus_label_file, foreground_label_file in zip(swi_list, thrombus_labels_list, foreground_labels_list):
        subject = HighToleranceSubject(
            swi_image=tio.ScalarImage(os.path.join(swi_dir, swi_file)),
            thrombus_label = tio.LabelMap(os.path.join(thrombus_labels_dir, thrombus_label_file)),
            foreground_label = tio.LabelMap(os.path.join(foreground_labels_dir, foreground_label_file)),
            subject_number = "_".join(swi_file.split("_")[:2])
        )
        subjects_list.append(subject)
    
    return tio.SubjectsDataset(subjects_list, **kwargs)

def load_subjectsdataset_2channel (swi_dir, tof_dir, thrombus_labels_dir, foreground_labels_dir, **kwargs):
    subjects_list = []
    swi_list = os.listdir(swi_dir)
    tof_list = os.listdir(tof_dir)
    thrombus_labels_list = os.listdir(thrombus_labels_dir)
    foreground_labels_list = os.listdir(foreground_labels_dir)
    
    if len(swi_list) != len(tof_list) != len(thrombus_labels_list) != len(foreground_labels_list):
        print("Mismatch in sample numbers")
    
    for swi_file, tof_file, thrombus_label_file, foreground_label_file in zip(swi_list, tof_list, thrombus_labels_list, foreground_labels_list):
        subject = HighToleranceSubject(
            swi_image=tio.ScalarImage(os.path.join(swi_dir, swi_file)),
            tof_image=tio.ScalarImage(os.path.join(tof_dir, tof_file)),
            thrombus_label = tio.LabelMap(os.path.join(thrombus_labels_dir, thrombus_label_file)),
            foreground_label = tio.LabelMap(os.path.join(foreground_labels_dir, foreground_label_file)),
            subject_number = "_".join(swi_file.split("_")[:2])
        )
        subjects_list.append(subject)
    
    return tio.SubjectsDataset(subjects_list, **kwargs)

# Inference Function

def run_inference (inference_dataloader, model_for_prediction, patch_size, patch_overlap, inference_dir_location, return_logits=True, patch_loader_batchsize=10, logit_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_for_prediction = model_for_prediction.to(device)

    whole_image_predictions = []

    for batch in inference_dataloader:
        for subject in batch:
            grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
            patch_loader = tio.SubjectsLoader(grid_sampler, batch_size=patch_loader_batchsize)
            aggregator = tio.inference.GridAggregator(grid_sampler)
            
            with torch.no_grad(): 
                for patches_batch in tqdm.tqdm(patch_loader, desc="Running Inference"):
                    input_tensor = patches_batch["swi_image"][tio.DATA].to(device)
                    locations = patches_batch[tio.LOCATION]
                    logits = model_for_prediction(input_tensor)[0]
                    logits = logits.cpu()
                    if return_logits == False:
                        labels = logit_to_binary_mask(logits, threshold=logit_threshold)
                        aggregator.add_batch(labels, locations)
                    else:
                        aggregator.add_batch(logits, locations)
            whole_prediction = aggregator.get_output_tensor().cpu()
            whole_image_predictions.append(whole_prediction)
    return restore_inference_original_size(whole_image_predictions, inference_dir_location)