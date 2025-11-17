import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

INPUT_SHAPE = (1, 28, 28, 28) # (C, D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 4
TIME_STEPS = 10 # Time series size 

def visualize_slices_inmodel(input_batch, target_batch, recon_batch, seg_batch):
    """
    Plots a 2D slice from the middle of a 4-tensor batch.
    """
    # Use torch.no_grad() to stop tracking gradients
    with torch.no_grad():
        slice_idx = input_batch.shape[2] // 2  # Middle slice index
        # --- 1. Process Tensors ---
        
        # Move to CPU and convert to NumPy
        # Squeeze out the channel dim (C=1)
        input_slice = input_batch.to('cpu').numpy()[0, 0, slice_idx, :, :]
        
        # Target is [B, D, H, W], no channel
        target_slice = target_batch.to('cpu').numpy()[0, slice_idx, :, :]
        
        # Recon needs detach() because it has grads
        recon_slice = recon_batch.to('cpu').detach().numpy()[0, 0, slice_idx, :, :]
        
        seg_logits = seg_batch.to('cpu').detach()
        seg_pred = torch.argmax(seg_logits, dim=1) # Shape: [B, D, H, W]
        seg_slice = seg_pred.numpy()[0, slice_idx, :, :]

        
        # --- 2. Plotting ---
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot 1: Original Input
        axes[0].imshow(input_slice, cmap='gray')
        axes[0].set_title(f"Input Image (Slice {slice_idx})")
        axes[0].axis('off')
        
        # Plot 2: Ground Truth Segmentation
        axes[1].imshow(target_slice, cmap='gray', vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Target Segmentation")
        axes[1].axis('off')
        
        # Plot 3: Reconstructed Output
        axes[2].imshow(recon_slice, cmap='gray')
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')
        
        # Plot 4: Predicted Segmentation
        axes[3].imshow(seg_slice, cmap='grey', vmin=0, vmax=NUM_CLASSES-1)
        axes[3].set_title("Predicted Segmentation")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()

def visualize_slices(input_batch, target_batch, recon_batch, seg_batch, slice_idx=None):
    """
    Plots a 2D slice from the middle of a 4-tensor batch.
    """
    # Use torch.no_grad() to stop tracking gradients
    with torch.no_grad():
        # --- 1. Process Tensors ---
        
        # Move to CPU and convert to NumPy
        # Squeeze out the channel dim (C=1)
        input_slice = input_batch.to('cpu').numpy()[0, 0, slice_idx, :, :]
        
        # Target is [B, D, H, W], no channel
        target_slice = target_batch.to('cpu').numpy()[0, slice_idx, :, :]
        
        # Recon needs detach() because it has grads
        recon_slice = recon_batch.to('cpu').detach().numpy()[0, 0, slice_idx, :, :]
        
        seg_logits = seg_batch.to('cpu').detach()
        seg_pred = torch.argmax(seg_logits, dim=1) # Shape: [B, D, H, W]
        seg_slice = seg_pred.numpy()[0, slice_idx, :, :]

        
        # --- 2. Plotting ---
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot 1: Original Input
        axes[0].imshow(input_slice, cmap='gray')
        axes[0].set_title(f"Input Image (Slice {slice_idx})")
        axes[0].axis('off')
        
        # Plot 2: Ground Truth Segmentation
        axes[1].imshow(target_slice, cmap='gray', vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Target Segmentation")
        axes[1].axis('off')
        
        # Plot 3: Reconstructed Output
        axes[2].imshow(recon_slice, cmap='gray')
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')
        
        # Plot 4: Predicted Segmentation
        axes[3].imshow(seg_slice, cmap='grey', vmin=0, vmax=NUM_CLASSES-1)
        axes[3].set_title("Predicted Segmentation")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Implements a multi-class Dice loss.
    
    Expects:
    - y_pred: Raw, unnormalized logits from the model
              Shape: [B, C, H, W] (or [B, C, D, H, W] for 3D)
    - y_true: Ground truth labels (integers)
              Shape: [B, 1, H, W] (or [B, 1, D, H, W] for 3D)
    """
    def __init__(self, num_classes, smooth=1e-6, include_background=False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        # By default, we calculate loss for all classes, including background.
        # If you want to ignore background, set include_background=False
        self.start_channel = 0 if include_background else 1

    def forward(self, y_pred, y_true):
        # 1. Apply Softmax to model output to get probabilities
        # y_pred_probs shape: [B, C, H, W]
        y_pred_probs = F.softmax(y_pred, dim=1)

        # 2. Convert ground truth labels to one-hot format
        # y_true_one_hot shape: [B, C, H, W]
        y_true_one_hot = F.one_hot(y_true.squeeze(1).long(), num_classes=self.num_classes)
        # Permute to match [B, C, H, W] or [B, C, D, H, W]
        # For 2D (B, H, W, C) -> (B, C, H, W)
        # For 3D (B, D, H, W, C) -> (B, C, D, H, W)
        dims = list(range(len(y_true_one_hot.shape)))
        dims.insert(1, dims.pop()) # Move last dim (C) to second dim
        y_true_one_hot = y_true_one_hot.permute(*dims).float()
        
        # 3. Flatten tensors but keep batch and class dims
        # Shape: [B, C, -1]
        y_pred_flat = y_pred_probs.view(y_pred_probs.shape[0], self.num_classes, -1)
        y_true_flat = y_true_one_hot.view(y_true_one_hot.shape[0], self.num_classes, -1)

        # 4. Calculate intersection and union per class (over the batch)
        # Sum over the last dim (pixels)
        intersection = (y_pred_flat * y_true_flat).sum(dim=2)
        union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)
        
        # 5. Calculate Dice score per class
        # Add smooth to avoid 0/0
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 6. Average Dice score across the specified classes (e.g., ignoring background)
        # We take the mean over the classes (dim=1) and then over the batch (dim=0)
        dice_loss = 1 - dice_per_class[:, self.start_channel:].mean()
        
        return dice_loss

class LiverDataset(Dataset):
    """
    Made by AI
    Custom PyTorch Dataset for the 3D Liver Segmentation data.
    """
    def __init__(self, image_dir, label_dir, target_size=(28, 28, 28)):
        
        # --- THIS IS THE CORRECTED PART (looking for .nii) ---
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.nii")))
        # --- END OF CORRECTION ---
        
        self.target_size = target_size # (D, H, W)
        
        # Ensure we have matched pairs
        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        assert len(self.label_paths) > 0, f"No labels found in {label_dir}"
        assert len(self.image_paths) == len(self.label_paths), \
            f"Found {len(self.image_paths)} images but {len(self.label_paths)} labels."
        
        print(f"Found {len(self.image_paths)} image/label pairs.")

    def __len__(self):
        return len(self.image_paths)

    def normalize(self, data):
        # Normalize pixel values to [0, 1]
        data = data - torch.min(data)
        data = data / torch.max(data)
        return data

    def __getitem__(self, idx):
        # 1. Load NIfTI files (nibabel handles .nii and .nii.gz the same way)
        img_nii = nib.load(self.image_paths[idx])
        lbl_nii = nib.load(self.label_paths[idx])
        
        # 2. Get data as numpy array and convert to tensor
        img_tensor = torch.from_numpy(img_nii.get_fdata()).float().permute(2, 1, 0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_nii.get_fdata()).long().permute(2, 1, 0).unsqueeze(0)

        # 3. Resize
        img_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                    size=self.target_size, 
                                    mode='trilinear', 
                                    align_corners=False).squeeze(0)
        
        lbl_resized = F.interpolate(lbl_tensor.float().unsqueeze(0), 
                                    size=self.target_size, 
                                    mode='nearest').squeeze(0).long()

        # 4. Normalize image
        img_resized = self.normalize(img_resized)

        # Squeeze the channel dim from the label
        lbl_resized = lbl_resized.squeeze(0) 

        return img_resized, lbl_resized
    
    