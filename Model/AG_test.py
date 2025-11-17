import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.optim import lr_scheduler 
import nibabel as nib
import os
import sys
import glob
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.utill import visualize_slices, DiceLoss
from func.Models import MultiTaskNet_ag as MultiTaskNet
from func.dataloads import LiverDataset_aug, LiverUnlabeledDataset_aug
from func.loss import BoundaryLoss, ComboLoss

INPUT_SHAPE = (128, 160, 160) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 
BATCH_SIZE = 1
VAL_SPLIT = 0.2  # 
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 0.001
NUM_EPOCHS = 300


SEG_WEIGHT = 4.0
RECON_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "./Task03_Liver_rs" 
data_root_folder = Path.cwd() / DATA_DIR

# --- 1. Load and Split Data ---
try:
    # Load the full labeled dataset
    full_labeled_dataset = LiverDataset_aug(image_dir=data_root_folder, label_dir=data_root_folder, target_size=INPUT_SHAPE)
    
    # Split into training and validation sets
    val_size = int(len(full_labeled_dataset) * VAL_SPLIT)
    train_size = len(full_labeled_dataset) - val_size
    train_dataset, val_dataset = random_split(full_labeled_dataset, [train_size, val_size])
    
    print(f"--- Data Split ---")
    print(f"Total labeled samples: {len(full_labeled_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoader for labeled training data
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # DataLoader for validation data
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE, # Can often be larger than train batch size
        shuffle=False # No need to shuffle validation data
    )

except Exception as e:
    print(f"Error creating Labeled dataset: {e}")
    exit()

try:
    unlabeled_dataset = LiverUnlabeledDataset_aug(
        image_dir=data_root_folder, 
        subfolder="imagesUnlabelledTr",
        target_size= INPUT_SHAPE,
        augment=True
    )
    
    # DataLoader for unlabeled data
    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    print("--- Unlabeled data success ---")

except Exception as e:
    print(f"Error creating Unlabeled dataset: {e}")
    exit()

# --- 2. Model, Losses, and Optimizer ---
if __name__ == "__main__":
    model = MultiTaskNet(
        in_channels=1, 
        num_classes=NUM_CLASSES, 
        latent_dim=LATENT_DIM  
    ).to(device)

    # Segmentation Loss (Dice + CE)
    loss_fn_seg_dice = DiceLoss(num_classes=NUM_CLASSES)
    loss_fn_seg_cross = nn.CrossEntropyLoss()
    loss_fn_seg = ComboLoss(
        dice_loss_fn=loss_fn_seg_dice,
        wce_loss_fn=loss_fn_seg_cross,
        alpha=0.6, beta=0.4  # You can tune these alpha/beta from func/loss.py
    ).to(device)
    
    # Reconstruction Loss
    loss_fn_recon = nn.MSELoss()

    optimizer_model = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- NEW: Learning Rate Scheduler ---
    # Monitors 'epoch_val_iou' and reduces LR if it stops improving
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode='max',      
        factor=0.1,      # Reduce LR by 90%
        patience=10,     # Wait 10 epochs of no improvement
        verbose=True
    )

    best_val_iou = 0.0
    SAVE_PATH = Path.cwd() / "Trained_models" / "AG_test.pth"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    
    print("--- Starting Training ---")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- TRAINING PHASE ---
        model.train() 
        
        # --- NEW: Epoch loss trackers ---
        epoch_train_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        
        # Use itertools.cycle to loop over the (likely shorter) unlabeled loader
        # Ensure train_loader is the one driving the epoch length
        for batch_idx, ((x_labeled, y_seg_target), (x_unlabeled)) in \
                enumerate(zip(train_loader, itertools.cycle(unlabeled_loader))):
            
            x_labeled = x_labeled.to(device)       
            y_seg_target = y_seg_target.to(device) 
            x_unlabeled = x_unlabeled[0].to(device) 
            
            optimizer_model.zero_grad()
            
            seg_out, recon_out_labeled = model(x_labeled)
            
            noise_factor = 0.1
            noise = torch.randn_like(x_unlabeled) * noise_factor
            x_unlabeled_noisy = x_unlabeled + noise
            _ , recon_out_unlabeled = model(x_unlabeled_noisy)
                        
            # Segmentation Loss (from labeled data)
            loss_seg = loss_fn_seg(seg_out, y_seg_target)

            # Reconstruction Loss (from both)
            loss_recon = loss_fn_recon(recon_out_labeled, x_labeled) + \
                               loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            # --- Final weighted total loss ---
            total_loss = (loss_seg * SEG_WEIGHT) + (loss_recon * RECON_WEIGHT) 
            
            total_loss.backward()
            optimizer_model.step()
            
            # --- NEW: Accumulate epoch losses ---
            epoch_train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += loss_recon.item()
    
            if batch_idx % 30 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Total Loss: {total_loss.item():.4f} | Recon Loss: {loss_recon.item():.4f} | Seg Loss: {loss_seg.item():.4f}")

        # --- NEW: Print average epoch training losses ---
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_seg_loss = epoch_seg_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        print(f"TRAIN Avg Loss: {avg_train_loss:.4f} | Avg Seg Loss: {avg_seg_loss:.4f} | Avg Recon Loss: {avg_recon_loss:.4f}")

        
        # --- NEW: VALIDATION PHASE ---
        model.eval()
        total_val_iou = 0.0
        
        with torch.no_grad():
            for (x_val, y_val_seg) in val_loader:
                x_val = x_val.to(device)
                y_val_seg = y_val_seg.to(device)
                
                # Get segmentation prediction
                val_seg_out, _ = model(x_val)
                
                # Calculate IoU for this batch
                # Convert logits (B, C, D, H, W) to class predictions (B, D, H, W)
                val_preds_classes = torch.argmax(val_seg_out, dim=1)
                y_val_seg_long = y_val_seg.squeeze(1).long() # Ensure target is (B, D, H, W) and long type

                batch_iou_list = []
                # Calculate IoU for each FOREGROUND class (skip background class 0)
                for c in range(1, NUM_CLASSES):
                    pred_c = (val_preds_classes == c)
                    true_c = (y_val_seg_long == c)
                    
                    intersection = (pred_c & true_c).sum()
                    union = (pred_c | true_c).sum()
                    
                    # Add epsilon for numerical stability (to avoid 0/0)
                    iou_c = (intersection.float() + 1e-6) / (union.float() + 1e-6)
                    batch_iou_list.append(iou_c.item())
                
                # Average IoU across foreground classes for this batch
                avg_batch_iou = np.mean(batch_iou_list)
                total_val_iou += avg_batch_iou
        
        # Average IoU for the entire validation set this epoch
        epoch_val_iou = total_val_iou / len(val_loader)
        print(f"VALIDATION | Epoch Mean IoU (mIoU): {epoch_val_iou:.4f}")

    print("--- Training Finished ---")
    print(f"Best validation mIoU achieved: {best_val_iou:.4f}")
    print(f"Best model saved to {SAVE_PATH}")
    torch.save(model.state_dict(), SAVE_PATH)