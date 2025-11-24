import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader
import random

# --- 1. Project Setup & Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import your custom model and dataset
# NOTE: Change 'VAE' to 'MultiTaskNet_ag' or 'MultiTaskNet_big' if testing those models
from func.Models import VAE, MultiTaskNet_ag, MultiTaskNet_big
from func.dataloaders import VolumetricPatchDataset 

# --- 2. Configuration ---
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4 
LATENT_DIM = 256
BATCH_SIZE = 1 

# DATA SPLIT: Use only the Test Set columns
# Training: 1-26, Validation: 27-32, Test: 33-38
TEST_COLUMNS = list(range(33, 39)) 

# Update this to point to the model you want to test
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "Trained_models" / "multi_big_final.pth"
SAVE_FILENAME = PROJECT_ROOT / "save_preds" / "multi_big_final.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- 3. Visualization Function ---
def visualize_and_save(x_tensor, y_tensor, seg_out, recon_out, save_path):
    """Visualizes the central slice of the 3D patch and saves it to disk."""
    slice_idx = x_tensor.shape[2] // 2

    # Convert to NumPy & Extract Slice
    x_np = x_tensor[0, 0, slice_idx, :, :].cpu().numpy()
    y_np = y_tensor[0, slice_idx, :, :].cpu().numpy()
    recon_np = recon_out[0, 0, slice_idx, :, :].cpu().detach().numpy()
    pred_seg_np = torch.argmax(seg_out, dim=1)[0, slice_idx, :, :].cpu().detach().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    titles = ['Input (Test Set)', 'Ground Truth', 'Reconstruction', 'Segmentation Pred']
    data = [x_np, y_np, recon_np, pred_seg_np]
    cmaps = ['gray', 'viridis', 'gray', 'viridis']
    
    for i, ax in enumerate(axes):
        vmax = NUM_CLASSES - 1 if i == 1 or i == 3 else None
        cax = ax.imshow(data[i], cmap=cmaps[i], vmin=0 if vmax else None, vmax=vmax)
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
        
        if i == 1 or i == 3:
            cbar = fig.colorbar(cax, ax=ax, ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f"Model Inference on Test Set Patch (Slice {slice_idx})", fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Visualization saved successfully to: {save_path}")


# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # A. Load the Model
    print(f"Loading model structure...")
    # UNCOMMENT the model you are testing
    #model = VAE(in_channels=1, latent_dim=LATENT_DIM, NUM_CLASSES=NUM_CLASSES).to(device)
    #model = MultiTaskNet_ag(in_channels=1, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(device)
    model = MultiTaskNet_big(in_channels=1, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(device)
    
    if MODEL_WEIGHTS_PATH.exists():
        print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"❌ Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        sys.exit(1)

    # B. Load One Data Patch from TEST SET
    print(f"Loading data from TEST columns: {TEST_COLUMNS}...")
    try:
        # Use selected_columns to restrict data to the Test Set
        test_dataset = VolumetricPatchDataset(
            selected_columns=TEST_COLUMNS, 
            augment=False, 
            is_labeled=True
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"Test set contains {len(test_dataset)} patches.")
        
        # Grab a single batch
        x_batch, y_batch = next(iter(test_loader))
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.squeeze(1).to(device)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # C. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # Check model output signature. MultiTaskNet_ag returns (seg, recon)
        # VAE returns (seg, recon, mu, logvar)
        output = model(x_batch)
        if len(output) == 2:
            seg_out, recon_out = output
        elif len(output) == 4: # VAE
            seg_out, recon_out, _, _ = output
        else:
            # Handle MultiTaskNet_big if it returns 3 values (seg, recon, latent)
            seg_out, recon_out = output[0], output[1]

    # D. Save Visualization
    save_path = Path.cwd() / SAVE_FILENAME
    visualize_and_save(x_batch, y_batch, seg_out, recon_out, save_path)

    # E. Calculate mIoU for this single patch
    print("Calculating IoU for this patch...")
    val_preds = torch.argmax(seg_out, dim=1)
    y_val_seg = y_batch 
    
    class_iou = []
    
    print("  [Class Metrics]")
    for c in range(NUM_CLASSES):
        pred_c = (val_preds == c)
        true_c = (y_val_seg == c)

        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0 # Or NaN if you prefer to exclude it
        
        class_iou.append(iou)
        print(f"    Class {c}: IoU = {iou:.4f} (Inter: {intersection}, Union: {union})")

    # Calculate mean IoU for foreground classes (1, 2, 3)
    foreground_iou = class_iou[1:]
    mIoU = np.mean(foreground_iou)
    print(f"  Mean IoU (Foreground): {mIoU:.4f}")