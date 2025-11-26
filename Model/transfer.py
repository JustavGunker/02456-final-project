import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import os
import sys
import itertools
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.loss import KLAnnealing, ComboLoss, FocalLoss, TverskyLoss, kld_loss
from func.Models import VAE
from func.dataloaders import TransferLabeledDataset, TransferUnlabeledDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LATENT_DIM = 256
NUM_EPOCHS = 100
BATCH_SIZE = 2 
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4 

# Lower learning rate for Transfer Learning to avoid destroying pre-trained features
LEARNING_RATE = 1e-5

SEG_WEIGHT = 10.0 # High weight to force adaptation to new labels
RECON_WEIGHT = 1.0

# --- PATHS ---
OUTPUT_DIR = PROJECT_ROOT / "outputs_VAE_transfer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the BEST model from the previous large-scale training
PRETRAINED_PATH = PROJECT_ROOT / "Trained_models" / "VAE_val_final.pth"

SAVE_PATH_BEST = PROJECT_ROOT / "Trained_models" / "VAE_transfer_best.pth"
SAVE_PATH_FINAL = PROJECT_ROOT / "Trained_models" / "VAE_transfer_final.pth"

# --- DATA CONFIGURATION ---
# Define columns for the UNLABELED background data (Source Domain)
# We use a subset or all of the original training columns to keep the VAE robust
# Using original training columns [1...26]
UNLABELED_COLS = list(range(1, 27)) 

def save_predictions_local(epoch, input_x, gt_y, recon_out, seg_out, output_dir, slice_idx=64):
    """Saves visualization of the transfer learning progress."""
    x_np = input_x[0, 0, slice_idx, :, :].cpu().numpy()
    
    # Handle cases where GT might be missing or different shape
    if gt_y is not None:
        y_np = gt_y[0, slice_idx, :, :].cpu().numpy()
    else:
        y_np = np.zeros_like(x_np)
        
    recon_np = recon_out[0, 0, slice_idx, :, :].cpu().detach().numpy()
    pred_seg_np = torch.argmax(seg_out, dim=1)[0, slice_idx, :, :].cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ['Input', 'Ground Truth', 'Recon', 'Seg Pred']
    data = [x_np, y_np, recon_np, pred_seg_np]
    
    for i, ax in enumerate(axes):
        cmap = 'viridis' if i == 1 or i == 3 else 'gray'
        vmax = NUM_CLASSES - 1 if i == 1 or i == 3 else None
        ax.imshow(data[i], cmap=cmap, vmin=0 if vmax else None, vmax=vmax)
        ax.set_title(titles[i])
        ax.axis('off')

    plt.suptitle(f"Transfer Epoch {epoch+1}")
    plt.savefig(output_dir / f"epoch_{epoch+1:04d}.png")
    plt.close(fig)

def plot_learning_curves(train_losses, train_ious, output_dir):
    """Plots Training Loss and Training mIoU curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Training Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Training mIoU
    if len(train_ious) > 0:
        ax2.plot(epochs, train_ious, 'g-', label='Training mIoU')
        ax2.set_title('Training Mean IoU')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "transfer_learning_curves.png")
    plt.close(fig)
    print(f"  Learning curves saved to: {output_dir / 'transfer_learning_curves.png'}")

if __name__ == "__main__":
    
    # --- 1. LOAD DATA ---
    try:
        print("Initializing Transfer Dataloaders...")
        
        # Labeled Train: The new small volume (slices 0-15)
        # Returns (1, 16, 128, 128) which model handles via adaptive pooling
        train_dataset = TransferLabeledDataset(mode='train', augment=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        # Note: Validation set removed as requested. 
        # Test data is implicitly held out by the 'mode=test' logic in the dataset class, 
        # but we are not loading it here.
        
        # Unlabeled Source: The original large dataset patches
        # Returns (1, 128, 128, 128)
        unlabeled_dataset = TransferUnlabeledDataset(selected_columns=UNLABELED_COLS, augment=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        print(f"--- Loaders Ready ---")
        print(f"Target Train Batches (New Data): {len(train_loader)}")
        print(f"Source Unlabeled Batches (Old Data): {len(unlabeled_loader)}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        exit()
    
    # --- 2. MODEL SETUP ---
    model = VAE(
        in_channels=1, 
        latent_dim=LATENT_DIM, 
        NUM_CLASSES=NUM_CLASSES
    ).to(DEVICE)
    
    # Load Pre-trained Weights
    if PRETRAINED_PATH.exists():
        print(f"Loading pre-trained weights from: {PRETRAINED_PATH}")
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"WARNING: Pre-trained weights not found at {PRETRAINED_PATH}. Starting from scratch.")

    # Loss Functions
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.5, beta=0.5).to(DEVICE)
    focal = FocalLoss(gamma=2.0).to(DEVICE)
    loss_fn_recon = nn.MSELoss().to(DEVICE)
    
    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=focal,
        alpha=0.5, 
        beta=0.5   
    ).to(DEVICE)

    optimizer_model = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

    # Scheduler tracking training metrics
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # No annealing for transfer learning, start with full KLD to maintain latent structure
    KLD_WEIGHT = 0.005 
    
    # --- 3. TRAINING LOOP ---
    best_train_iou = 0.0
    SAVE_INTERVAL = 5 # Save more frequently for transfer learning
    
    train_loss_history = []
    train_iou_history = []

    print("--- Starting Transfer Learning ---")

    # We cycle the UNLABELED loader because it is much larger than the labeled target data
    unlabeled_iterator = itertools.cycle(unlabeled_loader)

    for epoch in range(NUM_EPOCHS):        
        
        model.train()
        train_loss = 0
        train_iou_accum = 0.0
        
        last_x, last_y, last_rec, last_seg = None, None, None, None

        # Iterate over the SMALL target dataset
        for batch_idx, (x, y_target) in enumerate(train_loader):
            
            # Get a batch of source data (unlabeled)
            x_unlabeled = next(unlabeled_iterator)
            
            x = x.to(DEVICE)
            y_target = y_target.to(DEVICE).squeeze(1) # Remove channel dim
            x_unlabeled = x_unlabeled.to(DEVICE)

            optimizer_model.zero_grad()

            # 1. Forward on Target (New Data) -> Seg + Recon
            seg_out, recon_out, z_mu, z_logvar = model(x)
            
            # 2. Forward on Source (Old Data) -> Recon Only (Regularization)
            # We add noise to force robust features
            noise = torch.randn_like(x_unlabeled) * 0.1 
            _, recon_unlabeled, mu_u, logvar_u = model(x_unlabeled + noise)

            # 3. Losses
            # Segmentation on New Data
            l_seg = loss_fn_seg(seg_out, y_target)
            
            # Reconstruction on Both (Maintains old features + learns new texture)
            l_recon = loss_fn_recon(recon_out, x) + loss_fn_recon(recon_unlabeled, x_unlabeled)
            
            # KL Divergence
            l_kld = kld_loss(z_mu, z_logvar) + kld_loss(mu_u, logvar_u)
            l_kld = l_kld / (x.size(0) + x_unlabeled.size(0))

            total_loss = (l_seg * SEG_WEIGHT) + \
                         (l_recon * RECON_WEIGHT) + \
                         (l_kld * KLD_WEIGHT)
            
            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()

            # --- Calculate Training mIoU for this batch ---
            preds = torch.argmax(seg_out, dim=1)
            batch_iou = 0.0
            for c in range(1, NUM_CLASSES):
                inter = ((preds == c) & (y_target == c)).sum().float()
                union = ((preds == c) | (y_target == c)).sum().float()
                batch_iou += (inter + 1e-6) / (union + 1e-6)
            train_iou_accum += (batch_iou / (NUM_CLASSES - 1)).item()

            if batch_idx == len(train_loader) - 1:
                last_x, last_y, last_rec, last_seg = x.detach(), y_target.detach(), recon_out.detach(), seg_out.detach()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou_accum / len(train_loader)
        
        train_loss_history.append(avg_train_loss)
        train_iou_history.append(avg_train_iou)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train mIoU: {avg_train_iou:.4f} | Best: {best_train_iou:.4f}")
        
        scheduler.step(avg_train_iou)
        
        if avg_train_iou > best_train_iou:
            best_train_iou = avg_train_iou
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            print(f"  --> New Best Model Saved (on Training Set)!")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            print("  Saving visualization...")
            # slice_idx=8 is middle of the target_depth=16 volume
            save_idx = last_x.shape[2] // 2
            save_predictions_local(epoch, last_x, last_y, last_rec, last_seg, OUTPUT_DIR, slice_idx=save_idx)

    torch.save(model.state_dict(), SAVE_PATH_FINAL)
    
    plot_learning_curves(train_loss_history, train_iou_history, OUTPUT_DIR)
    
    print("Done.")