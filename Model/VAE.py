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


from func.dataloaders import VolumetricPatchDataset 
from func.utill import save_predictions
from func.loss import DiceLoss, KLAnnealing, ComboLoss, FocalLoss, TverskyLoss, kld_loss
from func.Models import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- GLOBAL CONFIGURATION (ADAPTED FOR PATCHES) ---
LATENT_DIM = 256
num_epochs = 400
KLD_WEIGHT = 0.0001
BATCH_SIZE = 2
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4 # 0, 1, 2, 3
# ----------------------------------------------------

# Data path is now managed internally by VolumetricPatchDataset to use $BLACKHOLE
data_root_folder = PROJECT_ROOT # Placeholder, actual path is in the Dataset class
OUTPUT_DIR = PROJECT_ROOT / "outputs_VAE"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving visualizations to: {OUTPUT_DIR}")

if __name__ != "__main__":
    print(f"Project root folder: {PROJECT_ROOT}")

try:
    # 1. Labeled Dataset: Instantiate the new patch-based class
    labeled_dataset = VolumetricPatchDataset(augment=True, is_labeled=True)
    
    # DataLoader for labeled data
    labeled_loader = DataLoader(
        dataset=labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # NOTE: Reduced num_workers since batch size is 1.
        num_workers=2 
    )
    print("--- Labeled DataLoader success (using patched data) ---")

except Exception as e:
    print(f"Error creating Labeled dataset: {e}")
    exit()

try:
    # 2. Unlabeled Dataset: Instantiate the new patch-based class
    unlabeled_dataset = VolumetricPatchDataset(augment=False, is_labeled=False)
    
    # DataLoader for unlabeled data
    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    print("--- Unlabeled DataLoader success (using patched data) ---")

except Exception as e:
    print(f"Error creating Unlabeled dataset: {e}")
    exit()

if __name__ == "__main__":
    # Initialize model, loss functions, and optimizer
    model = VAE(
        in_channels=1, 
        latent_dim=LATENT_DIM, 
        NUM_CLASSES=NUM_CLASSES
    ).to(device)
    
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.6, beta=0.4).to(device)
    focal = FocalLoss(gamma=2.0).to(device)

    loss_fn_recon = nn.MSELoss()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=focal,
        alpha=0.4, 
        beta=0.6).to(device)

    KLD_Annealing_start = 0
    KLD_Annealing_end   = 20
    kl_scheduler = KLAnnealing(
        start_epoch=KLD_Annealing_start,
        end_epoch=KLD_Annealing_end,
        start_beta=0.0,
        end_beta=0.05)
    
    SAVE_INTERVAL = 20 

    print("--- Training the VAE on Patched Volumetric Data ---")

    for epoch in range(num_epochs):        
        KLD_WEIGHT = kl_scheduler.get_beta(epoch)

        model.train()
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        
        # NOTE: The loop now iterates over patched data (C, D=64, H=64, W=64)
        for batch_idx, ((x, y_target), x_unlabeled) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            # x_unlabeled comes from DataLoader as a batch tensor (not a tuple wrapper)
            
            x = x.to(device)
            y_target = y_target.squeeze(1).to(device)
            x_unlabeled = x_unlabeled.to(device)

            optimizer_model.zero_grad()

            seg_out, recon_out_labeled, z_mu, z_logvar = model(x)
            noise = torch.randn_like(x_unlabeled)*0.2
            _ , recon_out_unlabeled, z_mu_unlabeled, z_logvar_unlabeled = model(x_unlabeled+noise)
            
            # seg loss
            loss_seg = loss_fn_seg(seg_out, y_target)

            # recon loss
            loss_recon = loss_fn_recon(recon_out_labeled, x) + loss_fn_recon(recon_out_unlabeled, x_unlabeled)

            # KL loss
            loss_kld_labeled = kld_loss(z_mu, z_logvar)
            loss_kld_unlabeled = kld_loss(z_mu_unlabeled, z_logvar_unlabeled)
            total_kld_loss = (loss_kld_labeled + loss_kld_unlabeled) / (x.size(0) + x_unlabeled.size(0))
            
            # total loss
            total_loss = (loss_seg * 5.0) + \
                        (loss_recon * 1.0) + \
                        (total_kld_loss * KLD_WEIGHT)
            
            total_loss.backward()
            optimizer_model.step()
            train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += loss_recon.item()
            epoch_kld_loss += total_kld_loss.item()
            
            if batch_idx == len(labeled_loader) - 1:
                last_x = x
                last_y = y_target
                last_recon = recon_out_labeled
                last_seg = seg_out

        
        # Removed: last_x, last_y, last_recon, last_seg assignments
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_predictions(epoch, last_x, last_y, last_recon, last_seg, OUTPUT_DIR, slice_idx=64)

        avg_train_loss = train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)
        avg_kld_loss = epoch_kld_loss / len(labeled_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_train_loss:.4f} | KLD Beta: {KLD_WEIGHT:.4f}")
        print(f"  Seg Loss: {avg_seg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | KLD Loss: {avg_kld_loss:.4f}")
        
        # Removed: Conditional save block

    print("Training complete.")

    cd = Path.cwd()
    save_path = cd / "Trained_models" / "VAE.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")