import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
import itertools
from pathlib import Path
from scipy.io import loadmat
from skimage.util import random_noise

# --- Custom Imports (Assuming these still exist in func/) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# NOTE: We keep these imports for your existing loss and model structure
from func.utill import save_predictions
from func.loss import DiceLoss, ComboLoss
from func.Models import MultiTaskNet_big
from func.dataloaders import VolumetricPatchDataset

BLACKHOLE_PATH = os.environ.get('BLACKHOLE', '.')


INPUT_SHAPE_PATCH = (128, 128, 128)
NUM_CLASSES = 4
LATENT_DIM = 256
BATCH_SIZE = 2
SAVE_INTERVAL = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

OUTPUT_DIR = PROJECT_ROOT / "output_big"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    # 1. Labeled Dataset
    labeled_dataset = VolumetricPatchDataset(augment=True, is_labeled=True)
    labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print("--- Labeled DataLoader created successfully ---")

    # 2. Unlabeled Dataset
    unlabeled_dataset = VolumetricPatchDataset(augment=False, is_labeled=False)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print("--- Unlabeled DataLoader created successfully ---")

except Exception as e:
    print(f"Error creating Datasets: {e}")
    exit()


if __name__ == "__main__":
    model = MultiTaskNet_big(in_channels=1, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(device)
    dice = DiceLoss(num_classes=NUM_CLASSES)
    cross = nn.CrossEntropyLoss(reduction='mean') 
    loss_seg = ComboLoss(dice_loss_fn=dice, wce_loss_fn=cross)
    loss_fn_recon = nn.MSELoss()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)

    print("--- Training the MultiTaskNet on Patched Volumetric Data ---")
    NUM_EPOCHS = 300

    unlabeled_iterator = itertools.cycle(unlabeled_loader)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train() 
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        for batch_idx, (x_labeled, y_seg_target) in enumerate(labeled_loader):
            
            x_unlabeled = next(unlabeled_iterator)
            
            # --- Data Transfer to Device ---
            x_labeled = x_labeled.to(device)
            y_seg_target = y_seg_target.squeeze(1).to(device)
            x_unlabeled = x_unlabeled.to(device) 

            optimizer_model.zero_grad()
            
            # 1. Forward Pass on Labeled Data
            seg_out, recon_out_labeled, _ = model(x_labeled)
            
            total_loss_seg = loss_seg(seg_out, y_seg_target)
            loss_recon_labeled = loss_fn_recon(recon_out_labeled, x_labeled)

            # 2. Forward Pass on Unlabeled Data
            noise_factor = 0.1
            noise = torch.randn_like(x_unlabeled) * noise_factor
            x_unlabeled_noisy = x_unlabeled + noise
            
            _ , recon_out_unlabeled, _ = model(x_unlabeled_noisy)
            
            loss_recon_unlabeled = loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            total_loss_recon = loss_recon_labeled + loss_recon_unlabeled
            total_loss = total_loss_seg + (total_loss_recon * 0.6) 
                
            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()
            epoch_seg_loss += total_loss_seg.item()
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
    print("Saving model weights...")

    SAVE_PATH = Path.cwd()/ "Trained_models" / "multi_big.pth"
    torch.save(model.state_dict(), SAVE_PATH)

    print(f"Model saved to {SAVE_PATH}")