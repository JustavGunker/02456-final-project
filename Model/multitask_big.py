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
from func.utill import visualize_slices, DiceLoss
from func.Models import MultiTaskNet_big
from func.dataloads import LiverDataset, LiverUnlabeledDataset


INPUT_SHAPE = (128, 128, 128) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


DATA_DIR = "./Task03_Liver_rs" 
# This one path points to the root directory (e.g., ./Task03_Liver_rs)
data_root_folder = Path.cwd() / DATA_DIR


try:
    # labeled set
    labeled_dataset = LiverDataset(image_dir=data_root_folder, label_dir=data_root_folder, target_size= INPUT_SHAPE)
    
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

    unlabeled_dataset = LiverUnlabeledDataset(
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
    # start model
    model = MultiTaskNet_big(
        in_channels=1, 
        num_classes=NUM_CLASSES, 
        latent_dim=LATENT_DIM  
    ).to(device)

    # define loss and optimizer
    loss_fn_seg_dice = DiceLoss(num_classes= NUM_CLASSES)
    loss_fn_seg_cross = nn.CrossEntropyLoss()
    loss_fn_recon = nn.MSELoss()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

    print("--- Training the MultiTaskNet on Liver Data ---")
    NUM_EPOCHS = 200

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train() 
        
        # iterate over both loaders
        for batch_idx, ((x_labeled, y_seg_target), (x_unlabeled)) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            # Move all data to device
            x_labeled = x_labeled.to(device)
            y_seg_target = y_seg_target.to(device)
            x_unlabeled = x_unlabeled[0].to(device)

            optimizer_model.zero_grad()
            
            # forward pass on labeled data for both seg and recon
            seg_out, recon_out_labeled, _ = model(x_labeled)
            
            # segmentation losses
            loss_seg_cross = loss_fn_seg_cross(seg_out, y_seg_target)
            
            total_loss_seg = loss_seg_cross * 1
            loss_seg_dice = 0.0
            
            # add dice loss if cross entropy is low enough
            if loss_seg_cross.item() < 0.6:
                dice_loss_component = loss_fn_seg_dice(seg_out, y_seg_target)
                total_loss_seg = total_loss_seg + (dice_loss_component * 1)
                loss_seg_dice = dice_loss_component.item()
                
            # labeled recon loss
            loss_recon_labeled = loss_fn_recon(recon_out_labeled, x_labeled)
                
            # Forward pass only on unlabeled data for recon
            _ , recon_out_unlabeled, _ = model(x_unlabeled)
            
            # unlabeled recon loss
            loss_recon_unlabeled = loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            # Total recon loss
            total_loss_recon = loss_recon_labeled + loss_recon_unlabeled
            
            # Total combined loss
            total_loss = total_loss_seg + (total_loss_recon * 0.5) 
                
            total_loss.backward()
            optimizer_model.step()
            
            # Udate Logging
            if batch_idx % 14 == 0:
                if loss_seg_cross.item() > 0.6:
                    print(f"Batch {batch_idx}/{len(labeled_loader)} | Total Loss: {total_loss.item():.4f} | Recon Loss (Total): {total_loss_recon.item():.4f} | CE Loss (Labeled): {loss_seg_cross.item():.4f} (Dice not active)")
                else:
                    print(f"Batch {batch_idx}/{len(labeled_loader)} | Total Loss: {total_loss.item():.4f} | Recon Loss (Total): {total_loss_recon.item():.4f} | CE Loss (Labeled): {loss_seg_cross.item():.4f} | DICE Loss (Labeled): {loss_seg_dice:.4f}")

print("--- Training Finished ---")
print("Saving model weights...")

SAVE_PATH = Path.cwd().parent / "Trained_models" / "multi_big.pth"
torch.save(model.state_dict(), SAVE_PATH)

print(f"Model saved to {SAVE_PATH}")