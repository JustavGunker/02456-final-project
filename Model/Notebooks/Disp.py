import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader

# --- 1. Project Setup & Imports ---
# Automatically find the project root to import custom modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


from func.Models import VAE, MultiTaskNet_ag, MultiTaskNet_big
from func.dataloaders import VolumetricPatchDataset

# --- 2. Configuration ---
# Must match the training configuration (128^3 patches)
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4 # 0=Background, 1=Segment 1, 2=Segment 2, etc.
LATENT_DIM = 256
BATCH_SIZE = 1 # Inference is done one patch at a time

# Path to your trained weights
# Update 'VAE.pth' to 'AG.pth' or 'multi_big.pth' as needed
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "Trained_models" / "multi_big.pth"
# Output filename (saved in the CURRENT directory)
SAVE_FILENAME = "inference_result.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- 3. Visualization Function ---
def visualize_and_save(x_tensor, y_tensor, seg_out, recon_out, save_path):
    """
    Visualizes the central slice of the 3D patch and saves it to disk.
    """
    # Calculate the middle slice index (e.g., index 64 for a depth of 128)
    slice_idx = x_tensor.shape[2] // 2

    # --- Prepare Data for Plotting (Convert to NumPy & Extract Slice) ---
    
    # Input X: (B, C, D, H, W) -> Extract slice -> (H, W)
    x_np = x_tensor[0, 0, slice_idx, :, :].cpu().numpy()
    
    # Ground Truth Y: (B, D, H, W) -> Extract slice -> (H, W)
    # Note: We ensure y_tensor has no channel dim before passing it here
    y_np = y_tensor[0, slice_idx, :, :].cpu().numpy()
    
    # Reconstruction: (B, C, D, H, W) -> Extract slice -> (H, W)
    recon_np = recon_out[0, 0, slice_idx, :, :].cpu().detach().numpy()
    
    # Segmentation Prediction: (B, Classes, D, H, W) -> Argmax -> (H, W)
    pred_seg_np = torch.argmax(seg_out, dim=1)[0, slice_idx, :, :].cpu().detach().numpy()
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    titles = ['Input Patch', 'Ground Truth', 'Reconstruction', 'Segmentation Pred']
    data = [x_np, y_np, recon_np, pred_seg_np]
    
    # Use 'gray' for raw images and 'viridis' (or 'nipy_spectral') for labels
    cmaps = ['gray', 'viridis', 'gray', 'viridis']
    
    for i, ax in enumerate(axes):
        # Set vmin/vmax for labels to ensure colors map correctly to classes 0-3
        vmax = NUM_CLASSES - 1 if i == 1 or i == 3 else None
        
        cax = ax.imshow(data[i], cmap=cmaps[i], vmin=0 if vmax else None, vmax=vmax)
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
        
        # Add colorbar for segmentation maps
        if i == 1 or i == 3:
            cbar = fig.colorbar(cax, ax=ax, ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f"Model Inference on {INPUT_SHAPE} Patch (Slice {slice_idx})", fontsize=16)
    
    # Save to the script's current execution directory
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Visualization saved successfully to: {save_path}")


# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # A. Load the Model
    print(f"Loading model structure (VAE)...")
    model = MultiTaskNet_big(in_channels=1, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    #model = VAE(in_channels=1, latent_dim=LATENT_DIM, NUM_CLASSES=NUM_CLASSES).to(device)
    
    if MODEL_WEIGHTS_PATH.exists():
        print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
        # Use weights_only=False if dealing with older PyTorch versions or complex pickles
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
        except:
            # Fallback for older saves
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
            
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"❌ Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        sys.exit(1)

    # B. Load One Data Patch
    print("Loading data...")
    try:
        # Use the existing patched dataset (no augmentation for testing)
        test_dataset = VolumetricPatchDataset(augment=False, is_labeled=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Grab a single batch
        x_batch, y_batch = next(iter(test_loader))
        
        # Move to device
        x_batch = x_batch.to(device)
        
        # Squeeze the channel dimension from Ground Truth (B, 1, D, H, W) -> (B, D, H, W)
        # This is critical for correct visualization indexing
        y_batch = y_batch.squeeze(1).to(device)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # C. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # VAE output signature: seg_out, recon_out, mu, logvar
        seg_out, recon_out,_ = model(x_batch)

    # D. Save Visualization
    save_path = Path.cwd() / SAVE_FILENAME
    visualize_and_save(x_batch, y_batch, seg_out, recon_out, save_path)