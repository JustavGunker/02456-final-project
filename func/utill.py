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
import torch.nn.functional as F
import nibabel as nib
import sys
import glob
import itertools

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


def visualize_slices_tr(input_batch, target_batch, recon_batch, seg_batch, threshold, slice_idx=None):
    """
    Plots a 2D slice from the middle of a 4-tensor batch.
    """
    with torch.no_grad():
        # --- 1. Process Tensors ---
        
        # Move to CPU
        input_slice = input_batch.to('cpu').numpy()[0, 0, slice_idx, :, :]
        target_slice = target_batch.to('cpu').numpy()[0, slice_idx, :, :]
        recon_slice = recon_batch.to('cpu').detach().numpy()[0, 0, slice_idx, :, :]
        
        seg_logits = seg_batch.to('cpu').detach()
        
        # Option A: Standard Argmax (What you have now)
        seg_pred_argmax = torch.argmax(seg_logits, dim=1)
        seg_slice_argmax = seg_pred_argmax.numpy()[0, slice_idx, :, :]

        # Option B: Thresholding for Class 1 (e.g., Liver)
        # 1. Apply Softmax to get probabilities (0 to 1)
        seg_probs = F.softmax(seg_logits, dim=1)
        # 2. Get the probability map for Class 1 (Foreground)
        # Shape: [B, H, W]
        prob_map_class1 = seg_probs[0, 1, slice_idx, :, :].numpy()
        
        # 3. Apply Threshold (e.g., > 0.5)
        threshold = threshold
        seg_slice_threshold = (prob_map_class1 > threshold).astype(int)

        
        # --- 2. Plotting ---
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5)) # Increased to 5 plots
        
        # Plot 1: Input
        axes[0].imshow(input_slice, cmap='gray')
        axes[0].set_title(f"Input (Slice {slice_idx})")
        axes[0].axis('off')
        
        # Plot 2: Ground Truth
        axes[1].imshow(target_slice, cmap='gray', vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Target")
        axes[1].axis('off')
        
        # Plot 3: Reconstruction
        axes[2].imshow(recon_slice, cmap='gray')
        axes[2].set_title("Reconstruction")
        axes[2].axis('off')
        
        # Plot 4: Prob Map (Confidence) - THIS IS VERY USEFUL
        # Shows exactly how confident the model is
        im4 = axes[3].imshow(prob_map_class1, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title("Liver Probability Map")
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
        
        # Plot 5: Thresholded Prediction
        axes[4].imshow(seg_slice_threshold, cmap='gray')
        axes[4].set_title(f"Prediction > {threshold}")
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.show()
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
    
def save_predictions(epoch, original, target, recon, seg_pred, save_dir):
    """
    Saves middle-slice visualizations for 3D medical data.
    """
    # 1. Detach and move to CPU
    # Shape: (B, C, D, H, W) -> We assume Batch=1
    img_data = original.detach().cpu().numpy()[0, 0]
    recon_data = recon.detach().cpu().numpy()[0, 0]
    target_data = target.detach().cpu().numpy()[0] # Assuming target is (B, D, H, W)
    
    # Process Segmentation: Apply Argmax to get class indices
    # seg_pred shape: (B, Classes, D, H, W) -> (D, H, W)
    seg_map = torch.argmax(seg_pred, dim=1).detach().cpu().numpy()[0]

    # 2. Select the Middle Slice (Depth dimension)
    slice_idx = img_data.shape[0] // 2

    # --- Save Reconstruction ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_data[slice_idx], cmap='gray')
    axes[0].set_title("Original Input (Mid-Slice)")
    axes[0].axis('off')
    
    axes[1].imshow(recon_data[slice_idx], cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"recon_iter{epoch+1}.png")
    plt.close(fig) # Close to free memory

    # --- Save Segmentation ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground Truth
    axes[0].imshow(target_data[slice_idx], cmap='tab10', interpolation='nearest')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(seg_map[slice_idx], cmap='tab10', interpolation='nearest')
    axes[1].set_title("Predicted Seg")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / f"seg_iter{epoch+1}.png")
    plt.close(fig)
    
    