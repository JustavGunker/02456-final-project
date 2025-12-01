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
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
import sys
import glob
import itertools

INPUT_SHAPE = (1, 28, 28, 28) # (C, D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 4
TIME_STEPS = 10 # Time series size 

def get_custom_colormap():
    """
    Creates a custom colormap matching the user's class definitions:
    0: Blue
    1: Red
    2: Yellow
    3: Turquoise
    """
    # Define colors in order of class ID (0, 1, 2, 3)
    colors = [
        'blue',      # Class 0
        'red',       # Class 1
        'yellow',    # Class 2
        'cyan'       # Class 3 (Turquoise)
    ]
    return ListedColormap(colors)

def save_predictions(epoch, input_x, gt_y, recon_out, seg_out, output_dir, slice_idx=64):
    """
    Saves a central 2D slice of the input, ground truth, reconstruction, and prediction.
    Inputs are expected to be Tensors.
    
    Args:
        epoch (int): Current epoch number.
        input_x (torch.Tensor): Input batch (B, C, D, H, W).
        gt_y (torch.Tensor): Ground truth batch (B, D, H, W).
        recon_out (torch.Tensor): Reconstruction output (B, C, D, H, W).
        seg_out (torch.Tensor): Segmentation logits (B, Num_Classes, D, H, W).
        output_dir (str or Path): Directory to save the image.
        slice_idx (int): Index of the slice to visualize along the depth axis.
    """
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Move tensors to CPU and convert to NumPy
    # Input X: (B, C=1, D, H, W) -> (H, W)
    x_np = input_x[0, 0, slice_idx, :, :].cpu().numpy() 
    
    # Ground Truth Y: (B, D, H, W) -> (H, W)
    # Note: Assuming gt_y is already squeezed of channel dim as per training loop
    y_np = gt_y[0, slice_idx, :, :].cpu().numpy()       
    
    # Recon Output: (B, C=1, D, H, W) -> (H, W)
    recon_np = recon_out[0, 0, slice_idx, :, :].cpu().detach().numpy()
    
    # Seg Output: (B, C_classes, D, H, W) -> Argmax -> (H, W)
    pred_seg_np = torch.argmax(seg_out, dim=1)[0, slice_idx, :, :].cpu().detach().numpy()
    
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    titles = [f'Input (E{epoch+1})', 'Ground Truth', 'Reconstruction', 'Segmentation Pred']
    data = [x_np, y_np, recon_np, pred_seg_np]
    
    # Custom colormap for segmentation masks
    seg_cmap = get_custom_colormap()
    
    # Define colormaps for each subplot
    # Input and Recon are grayscale. GT and Pred use custom map.
    cmaps = ['gray', seg_cmap, 'gray', seg_cmap]
    
    # Number of classes for normalization of the discrete colormap
    num_classes = 4

    for i, ax in enumerate(axes):
        # Determine vmin/vmax based on type of data
        if i == 1 or i == 3: # Segmentation masks
            vmax = num_classes - 1
            vmin = 0
            interpolation = 'nearest' # Best for discrete labels
        else: # Grayscale images
            vmax = None
            vmin = None
            interpolation = None 
        
        im = ax.imshow(data[i], cmap=cmaps[i], vmin=vmin, vmax=vmax, interpolation=interpolation)
        ax.set_title(titles[i])
        ax.axis('off')
        
        # Add color bar specifically for labels and predictions to show class mapping
        if i == 1 or i == 3:
            cbar = fig.colorbar(im, ax=ax, ticks=range(num_classes), fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['Water', 'Oil', 'Solids', 'Gas'])

    filename = f"epoch_{epoch+1:04d}_predictions.png"
    save_path = output_dir / filename
    
    plt.suptitle(f"Epoch {epoch+1} Visualization (Patch Center Slice {slice_idx})", fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Visuals saved to: {save_path}")

def plot_learning_curves(train_losses, val_losses, val_ious, output_dir):
    """Plots Training/Validation Loss and mIoU curves."""
    
    # Check if we actually have data to plot
    if len(train_losses) == 0:
        print("No training data to plot.")
        return

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    # Only plot validation loss if we have the same number of data points
    if len(val_losses) == len(epochs):
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot mIoU
    if len(val_ious) > 0:
        ax2.plot(epochs, val_ious, 'g-', label='Validation mIoU')
        ax2.set_title('Validation Mean IoU')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    
    # FIX: Construct a full file path including the filename
    save_path = output_dir / "AG_learning_curve.png"
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Learning curves saved to: {save_path}")