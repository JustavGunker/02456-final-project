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

from func.Models import AutoencoderNet
from func.dataloads import LiverDataset_aug, LiverUnlabeledDataset_aug
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

INPUT_SHAPE = (232, 232, 128) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
LATENT_DIM = 256 # RNN batch
BATCH_SIZE = 2

DATA_DIR = "./Task03_Liver_rs" 
data_root_folder = Path.cwd() / DATA_DIR

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
        target_size= INPUT_SHAPE,
        augment=True
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
    
    model = AutoencoderNet(in_channels=1).to(device)

    loss_fn_recon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_batches = len(labeled_loader) + len(unlabeled_loader)
    
    NUM_PRETRAIN_EPOCHS = 150
    
    print("Pre-training Autoencoder")

    for epoch in range(NUM_PRETRAIN_EPOCHS):
        print(f"\n--- Pre-train Epoch {epoch+1}/{NUM_PRETRAIN_EPOCHS} ---")
        model.train()
        epoch_loss = 0
        combined_loader = itertools.chain(labeled_loader, unlabeled_loader)

        for batch_idx, data_batch in enumerate(combined_loader):
            x = data_batch[0].to(device)
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            recon_out = model(x)
            
            loss = loss_fn_recon(recon_out, x)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 35 == 0:
                print(f"Batch {batch_idx} | Recon Loss: {loss.item():.4f}")

            epoch_loss += loss.item()
        avg_loss = epoch_loss/num_batches
        print(f"Avg Epoch loss: {avg_loss:.4f}")

    print("--- Pre-training Finished ---")

    # Save the encoder weights
    SAVE_PATH = Path.cwd()/ "Trained_models" / "pretrained_ae_encoder.pth"
    print(f"Saving encoder weights to {SAVE_PATH}")
    torch.save(model.encoder.state_dict(), SAVE_PATH)
   