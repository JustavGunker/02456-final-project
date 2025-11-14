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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from func.utill import visualize_slices, visualize_slices_inmodel
from func.Models import SegmentationNet
from func.dataloads import LiverDataset, LiverUnlabeledDataset
from func.loss import BoundaryLoss, ExpLogComboLoss, DiceLoss

INPUT_SHAPE = (28, 28, 28) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 4

DATA_DIR = "./Task03_Liver_rs" 
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

if __name__ == "__main__":
    
    model = SegmentationNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    
    ENCODER_WEIGHTS_PATH = "Trained_models/encoder_weights_pretrained.pth"
    
    # Load the pre-trained encoder weights
    model.encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_PATH, map_location=device))
    print(f"Loaded pre-trained encoder weights from {ENCODER_WEIGHTS_PATH}")

    # freeze encoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder weights are FROZEN.")
    
    loss_fn_seg_dice = DiceLoss(num_classes= NUM_CLASSES)
    loss_fn_seg_cross = nn.CrossEntropyLoss()
    loss_fn_recon = nn.MSELoss()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

    # loss stage 1 
    loss_fn_seg_stage1 = ExpLogComboLoss(
        dice_loss_fn=loss_fn_seg_dice,
        wce_loss_fn=loss_fn_seg_cross,
        alpha=1.0, 
        beta=1.0, 
        gamma_dice=1.0,  # Standard Dice
        gamma_wce=1.0    # Standard CE
    ).to(device)

    # loss stage 2
    #loss_fn_seg_stage2 = ExpLogComboLoss(
    #    dice_loss_fn=loss_fn_seg_dice,
    #    wce_loss_fn=loss_fn_seg_cross,
    #    alpha=1.0, 
    #    beta=1.0, 
    #    gamma_dice=1.5,  # Focus on hard Dice
    #    gamma_wce=1.5    # Focus on hard CE
    #).to(device)
    # loss stage 2 - Boundary loss hausdorff
    loss_fn_seg_stage2 = BoundaryLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    NUM_FINETUNE_EPOCHS = 100
    BOUNDARY_FOCUS_EPOCH = 80
    
    print("--- PHASE 2: Fine-tuning Segmentation Decoder ---")
    
    for epoch in range(NUM_FINETUNE_EPOCHS):
        print(f"\n--- Finetune Epoch {epoch+1}/{NUM_FINETUNE_EPOCHS} ---")
        
        # --- Staged Loss Logic ---
        if epoch + 1 < BOUNDARY_FOCUS_EPOCH:
            current_seg_loss_fn = loss_fn_seg_stage1
            if epoch == 0: print("Using Stage 1 (Region) Loss")
        else:
            current_seg_loss_fn = loss_fn_seg_stage2
            if epoch + 1 == BOUNDARY_FOCUS_EPOCH: 
                print("--- SWITCHING TO STAGE 2 (BOUNDARY) LOSS ---")
        
        model.train()
        
        # --- We ONLY use the labeled_loader ---
        for batch_idx, (x_labeled, y_seg_target) in enumerate(labeled_loader):
            
            x_labeled = x_labeled.to(device)       
            y_seg_target = y_seg_target.to(device) 
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            seg_out = model(x_labeled)
            
            # --- Loss Calculation ---
            loss = current_seg_loss_fn(seg_out, y_seg_target)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(labeled_loader)} | Seg Loss: {loss.item():.4f}")

    print("--- Fine-tuning Finished ---")
    
    # save model
    SAVE_PATH = "Trained_models/final_segmentation_model.pth"
    # save both encoder and decoder weights
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Final segmentation model saved to {SAVE_PATH}")