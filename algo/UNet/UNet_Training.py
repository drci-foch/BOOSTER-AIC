import torch
import torchio as tio
import os
import json
import matplotlib.pyplot as plt
import tqdm

# Helper Functions
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    # Saves a checkpoint of a PyTorch model.
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    folder_path = os.path.dirname(filepath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    torch.save(checkpoint, filepath)

def get_image_set_breakdown (train_dir, val_dir, test_dir):
        # Returns a list of image number included in training, validation and test set for future reference.
    train_list = ["_".join(img_filename.split("_")[:2]) for img_filename in os.listdir(train_dir) if (img_filename.endswith(".nii.gz"))]
    val_list = ["_".join(img_filename.split("_")[:2]) for img_filename in os.listdir(val_dir) if (img_filename.endswith(".nii.gz"))]
    test_list = ["_".join(img_filename.split("_")[:2]) for img_filename in os.listdir(test_dir) if (img_filename.endswith(".nii.gz"))]
    return train_list, val_list, test_list    

def save_model_params (save_location, train_swi_dir, val_swi_dir, test_swi_dir,
                       patches_per_image, patch_size, queue_length, batch_size,
                       model_architecture, channels,
                       model_loss, model_optimizer,
                       model_notes=None, loss_notes=None, optimizer_notes=None, training_epochs=None):
    
    train_list, val_list, test_list = get_image_set_breakdown (train_swi_dir, val_swi_dir, test_swi_dir)
    
    model_params = {"train_total":len(train_list), "validation_total":len(val_list), "test_total": len(test_list), "patches_per_image":patches_per_image,
                "patch_size":patch_size, "queue_length":queue_length, "batch_size":batch_size,
                "model_architecture":model_architecture, "channels":channels, "model_notes":model_notes,
                "model_loss":model_loss, "loss_notes":loss_notes,
                "model_optimizer":model_optimizer, "optimizer_notes":optimizer_notes,
                "training_epochs":training_epochs,
                "train_image_numbers":train_list, "validation_image_numbers":val_list, "test_image_numbers":test_list
                }
    folder_path = os.path.dirname(save_location)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    with open(save_location, "w") as f:
        json.dump(model_params, f, ensure_ascii=False, indent=4)

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

# Training Function

def train_model (model, loss, optimizer, train_patches_loader, val_patches_loader, num_epochs=10, starting_epoch=1, save_checkpoint_flag=False, load_from_checkpoint=False, save_checkpoint_location=None, load_checkpoint_location=None, display_loss=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_from_checkpoint:
        model_checkpoint = torch.load(load_checkpoint_location, map_location=device, weights_only=False)
        model.load_state_dict(model_checkpoint["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])
    else:
        model.to(device)
    
    if display_loss:
        train_loss_history = []
        val_loss_history = []
    
    for epoch in range(starting_epoch, num_epochs+1):
        model.train()
        train_loss_value = 0
        for patches_batch in tqdm.tqdm(train_patches_loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
            images = patches_batch["swi_image"][tio.DATA]
            gt_masks = patches_batch["thrombus_label"][tio.DATA]
            images, gt_masks = images.to(device), gt_masks.to(device)
            
            optimizer.zero_grad()
            predicted_mask = model(images)[0]
            train_loss = loss(predicted_mask, gt_masks)
            train_loss.backward()
            optimizer.step()
            train_loss_value += train_loss.item()
        
        train_loss_value /= len(train_patches_loader)
        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss_value:.8f}")
        
        model.eval()
        val_loss_value = 0
        with torch.no_grad():
            for patches_batch in tqdm.tqdm(val_patches_loader, desc=f"Validating Epoch {epoch}/{num_epochs}"):
                images = patches_batch["swi_image"][tio.DATA]
                gt_masks = patches_batch["thrombus_label"][tio.DATA]
                images, gt_masks = images.to(device), gt_masks.to(device)
                
                predicted_mask = model(images)[0]
                val_loss = loss(predicted_mask, gt_masks)
                val_loss_value += val_loss.item()

        val_loss_value /= len(val_patches_loader)
        print(f"Epoch {epoch}/{num_epochs}, Validation Loss: {val_loss_value:.8f}")

        if display_loss:
            train_loss_history.append(train_loss_value)
            val_loss_history.append(val_loss_value)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
            ax.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss history")
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(0,0))
            ax.legend()
            plt.show()
        
        if save_checkpoint_flag:
            save_checkpoint(model, optimizer, epoch, loss, save_checkpoint_location)