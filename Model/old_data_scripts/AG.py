import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
import sys
import glob
import itertools
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.utill import visualize_slices, save_predictions
from func.Models import MultiTaskNet_ag as MultiTaskNet
from func.dataloads import LiverDataset_aug, LiverUnlabeledDataset_aug
from func.loss import BoundaryLoss, ComboLoss, TverskyLoss, DiceLoss

INPUT_SHAPE = (128, 160, 160) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "./Task03_Liver_rs" 
data_root_folder = PROJECT_ROOT / DATA_DIR
OUTPUT_DIR = PROJECT_ROOT / "output_AG "
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    # labeled set
    labeled_dataset = LiverDataset_aug(image_dir=data_root_folder, label_dir=data_root_folder, target_size= INPUT_SHAPE)
    
    #DataLoader for labeled data
    labeled_loader = DataLoader(
        dataset=labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    print("--- success ---")

except Exception as e:
    print(f"Error creating Labeled dataset: {e}")
    exit()

try:

    unlabeled_dataset = LiverUnlabeledDataset_aug(
        image_dir=data_root_folder, 
        subfolder="imagesUnlabelledTr",
        target_size= INPUT_SHAPE
    )
    
    # 
    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    print("--- success ---")

except Exception as e:
    print(f"Error creating Unlabeled dataset: {e}")
    exit()

if __name__ == "__main__":
    model = MultiTaskNet(
        in_channels=1, 
        num_classes=NUM_CLASSES, 
        latent_dim=LATENT_DIM  
    ).to(device)

    #loss_fn_seg_dice = DiceLoss(num_classes=NUM_CLASSES)
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.8, beta=0.2)
    loss_fn_seg_cross = nn.CrossEntropyLoss()
    loss_fn_recon = nn.MSELoss()

    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=loss_fn_seg_cross,
        alpha=0.4, beta=0.6
    ).to(device)
    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

    NUM_EPOCHS = 400
    SAVE_INTERVAL = 20
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train() 
        
        # Use itertools.cycle to loop over the (likely shorter) unlabeled loader
        for batch_idx, ((x_labeled, y_seg_target), (x_unlabeled)) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x_labeled = x_labeled.to(device)       
            y_seg_target = y_seg_target.to(device) 
            x_unlabeled = x_unlabeled[0].to(device) 
            
            optimizer_model.zero_grad()
            
            seg_out, recon_out_labeled = model(x_labeled)
            
            # We only care about the recon_out and latent_z here
            noise_factor = 0.1
            noise = torch.randn_like(x_unlabeled) * noise_factor
            x_unlabeled_noisy = x_unlabeled + noise
            _ , recon_out_unlabeled = model(x_unlabeled_noisy)
                        
            # Segmentation Loss (from labeled data, using the current stage's loss)
            loss_seg = loss_fn_seg(seg_out, y_seg_target)

            # Reconstruction Loss (from both)
            total_loss_recon = loss_fn_recon(recon_out_labeled, x_labeled) + loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            # Final total loss 
            total_loss = (loss_seg * 5) + (total_loss_recon * 1) 
            
            total_loss.backward()
            optimizer_model.step()

            last_x = x_labeled
            last_y = y_seg_target
            last_recon = recon_out_labeled
            last_seg = seg_out
    
            # This was your TypeError (printing the function, not the value)
            if batch_idx % 30 == 0:
                print(f"Batch {batch_idx}/{len(labeled_loader)} | Total Loss: {total_loss.item():.4f} | Recon Loss: {total_loss_recon.item():.4f} | Seg Loss: {loss_seg.item():.4f}")
        if (epoch == 0) or ((epoch + 1) % SAVE_INTERVAL == 0) or (epoch == NUM_EPOCHS - 1):
            print(f"  Saving visuals for Epoch {epoch+1}...")
            save_predictions(epoch, last_x, last_y, last_recon, last_seg, OUTPUT_DIR)

    print("--- Training Finished ---")
    
    SAVE_PATH = Path.cwd() / "Trained_models" / "AG.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
