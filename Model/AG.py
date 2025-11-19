import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import itertools
from pathlib import Path


# --- PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.utill import save_predictions
from func.dataloaders import VolumetricPatchDataset 
from func.loss import BoundaryLoss, ComboLoss, TverskyLoss, DiceLoss
from func.Models import MultiTaskNet_ag as MultiTaskNet 

# --- GLOBAL CONFIGURATION (ADAPTED FOR PATCHES) ---
# FIX: The model input shape is now the patch shape (D, H, W)
INPUT_SHAPE = (128,128,128) 
NUM_CLASSES = 4  # (Classes 0, 1, 2, 3)
LATENT_DIM = 256 
BATCH_SIZE = 3
SAVE_INTERVAL = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data path is now managed internally by VolumetricPatchDataset to use $BLACKHOLE
OUTPUT_DIR = PROJECT_ROOT / "output_AG "
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    # 1. Labeled Dataset: Instantiate the new patch-based class
    labeled_dataset = VolumetricPatchDataset(augment=True, is_labeled=True)
    
    labeled_loader = DataLoader(
        dataset=labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    print("--- Labeled DataLoader success (using patched data) ---")

except Exception as e:
    print(f"Error creating Labeled dataset: {e}")
    exit()

try:
    # 2. Unlabeled Dataset: Instantiate the new patch-based class
    unlabeled_dataset = VolumetricPatchDataset(
        augment=False, 
        is_labeled=False
    )
    
    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )
    print("--- Unlabeled DataLoader success (using patched data) ---")

except Exception as e:
    print(f"Error creating Unlabeled dataset: {e}")
    exit()

if __name__ == "__main__":
    # Initialize the model using the alias
    model = MultiTaskNet(
        in_channels=1, 
        num_classes=NUM_CLASSES, 
        latent_dim=LATENT_DIM  
    ).to(device)

    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.6, beta=0.4)
    loss_fn_seg_cross = nn.CrossEntropyLoss()
    loss_fn_recon = nn.MSELoss()

    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=loss_fn_seg_cross,
        alpha=0.4, beta=0.6
    ).to(device)
    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

    NUM_EPOCHS = 400
  

    print("--- Training the MultiTaskNet on Patched Volumetric Data ---")
    
    unlabeled_iterator = itertools.cycle(unlabeled_loader)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        model.train() 
        
        for batch_idx, ((x_labeled, y_seg_target), x_unlabeled) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x_labeled = x_labeled.to(device)       
            # Segmentation target needs (B, D, H, W) for CrossEntropy/ComboLoss
            y_seg_target = y_seg_target.squeeze(1).to(device) 
            x_unlabeled = x_unlabeled.to(device)
            
            optimizer_model.zero_grad()
            
            # Forward pass on Labeled data
            seg_out, recon_out_labeled = model(x_labeled)
            
            # Forward pass on unlabeled noisy data
            noise_factor = 0.1
            noise = torch.randn_like(x_unlabeled) * noise_factor
            x_unlabeled_noisy = x_unlabeled + noise
            _ , recon_out_unlabeled = model(x_unlabeled_noisy)
                        
            # Segmentation Loss 
            loss_seg = loss_fn_seg(seg_out, y_seg_target)

            # Reconstruction Loss
            total_loss_recon = loss_fn_recon(recon_out_labeled, x_labeled) + loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            # Final total loss 
            total_loss = (loss_seg * 5) + (total_loss_recon * 1) 
            

            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += total_loss_recon.item()
            if batch_idx == len(labeled_loader) - 1:
                last_x = x_labeled
                last_y = y_seg_target
                last_recon = recon_out_labeled
                last_seg = seg_out

        avg_train_loss = train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Seg Loss: {avg_seg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")
                # --- Visualization and Checkpoint Saving ---
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_predictions(epoch, last_x, last_y, last_recon, last_seg, OUTPUT_DIR, slice_idx=64)


    print("--- Training Finished ---")
    
    SAVE_PATH = Path.cwd() / "Trained_models" / "AG.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")