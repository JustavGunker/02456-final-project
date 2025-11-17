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
from func.loss import DiceLoss, KLAnnealing, ComboLoss, FocalLoss
from func.Models import VAE
from func.dataloads import LiverDataset_aug, LiverUnlabeledDataset_aug 

from func.loss import kld_loss
from xml.parsers.expat import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LATENT_DIM = 256
NUM_EPOCHS = 300
BATCH_SIZE = 1
INPUT_SHAPE = (128, 160, 160) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2
VAL_SPLIT = 0.2 # < 20% Vali
LEARNING_RATE = 1e-4

SEG_WEIGHT = 4
RECON_WEIGHT = 1

DATA_DIR = "./Task03_Liver_rs"
data_root_folder = Path.cwd() / DATA_DIR

if __name__ == "__main__":
    
    try:
        # Load the full labeled dataset
        full_labeled_dataset = LiverDataset_aug(image_dir=data_root_folder, label_dir=data_root_folder, target_size=INPUT_SHAPE, 
                                            augment=False) # Load with augment=False
        
        # Split into training and validation sets
        val_size = int(len(full_labeled_dataset) * VAL_SPLIT)
        train_size = len(full_labeled_dataset) - val_size
        train_dataset, val_dataset = random_split(full_labeled_dataset, [train_size, val_size])
        train_dataset.dataset.augment = True
        val_dataset.dataset.augment = False

        print(f"--- Data Split ---")
        print(f"Total labeled samples: {len(full_labeled_dataset)}")
        print(f"Training samples: {len(train_dataset)} (Augmentation: {train_dataset.dataset.augment})")
        print(f"Validation samples: {len(val_dataset)} (Augmentation: {val_dataset.dataset.augment})")
        
        # new data loader with data augmentation
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        print("--- Labeled data success ---")
    
    except Exception as e:
        print(f"Error creating Labeled dataset: {e}")
        exit()

    try:
        # unlabeled set
        unlabeled_dataset = LiverUnlabeledDataset_aug(
            image_dir=data_root_folder, 
            subfolder="imagesUnlabelledTr",
            target_size= INPUT_SHAPE,
            augment=True 
        )
        
        #DataLoader for unlabeled data
        unlabeled_loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        print("--- Unlabeled data success ---")
    except Exception as e:
        print(f"Error creating Unlabeled dataset: {e}")
        exit()
    
    model = VAE(
        in_channels=1, 
        latent_dim=LATENT_DIM, 
        NUM_CLASSES=NUM_CLASSES).to(device)
    
    dice = DiceLoss(num_classes=NUM_CLASSES)
    focal = FocalLoss(gamma=2.0).to(device) # Use Focal Loss
    loss_fn_recon = nn.MSELoss()
    
    loss_fn_seg = ComboLoss(
        dice_loss_fn=dice,
        wce_loss_fn=focal, # Pass Focal Loss
        alpha=0.5, 
        beta=0.5   
    ).to(device)

    optimizer_model = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Added weight_decay

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode='max',      # We want to MAXIMIZE IoU
        factor=0.1,      # Reduce LR by 90%
        patience=10,     # Wait 20 epochs of no improvement
        verbose=True
    )
    
    # KLD Annealing Scheduler (your corrected version)
    KLD_Annealing_start = 0
    KLD_Annealing_end   = 20 
    kl_scheduler = KLAnnealing(
        start_epoch=KLD_Annealing_start,
        end_epoch=KLD_Annealing_end,
        start_beta=0.0,
        end_beta=0.05) # Anneal to a small value
    
    
    best_val_iou = 0.0
    patience = 70 
    epochs_no_improve = 0
    SAVE_PATH = Path.cwd() / "Trained_models" / "VAE_test.pth"
    SAVE_PATH_IoU = Path.cwd() / "Trained_models" / "VAE_IoU.pth"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    
    print("--- Training the VAE on Liver Data ---")

    for epoch in range(NUM_EPOCHS):        
        KLD_WEIGHT = kl_scheduler.get_beta(epoch)
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} | KLD Beta: {KLD_WEIGHT:.6f} ---")

        model.train()
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        
        for batch_idx, ((x, y_target), (x_unlabeled)) in \
                enumerate(zip(train_loader, itertools.cycle(unlabeled_loader))):
            
            x = x.to(device)
            y_target = y_target.to(device)
            x_unlabeled = x_unlabeled[0].to(device)

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

            kld_per_pixel = (loss_kld_labeled + loss_kld_unlabeled) / (x.size(0) * x.numel())
            
            # total loss
            total_loss = (loss_seg * SEG_WEIGHT) + \
                        (loss_recon * RECON_WEIGHT) + \
                        (kld_per_pixel * KLD_WEIGHT) 
            
            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += loss_recon.item()
            epoch_kld_loss += kld_per_pixel.item() # Log the scaled KLD

        avg_train_loss = train_loss / len(train_loader)
        avg_seg_loss = epoch_seg_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kld_loss = epoch_kld_loss / len(train_loader)
        
        print(f"  TRAIN | Total Loss: {avg_train_loss:.4f}")
        print(f"          Seg: {avg_seg_loss:.4f} | Recon: {avg_recon_loss:.4f} | KLD: {avg_kld_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_iou = 0.0
        
        with torch.no_grad():
            for (x_val, y_val_seg) in val_loader:
                x_val = x_val.to(device)
                y_val_seg = y_val_seg.to(device)
                
                # Get segmentation prediction
                val_seg_out, _, _, _ = model(x_val)
                
                # Calculate IoU for this batch
                val_preds_classes = torch.argmax(val_seg_out, dim=1)
                y_val_seg_long = y_val_seg.squeeze(1).long()

                batch_iou_list = []
                for c in range(1, NUM_CLASSES): # Skip background class 0
                    pred_c = (val_preds_classes == c)
                    true_c = (y_val_seg_long == c)
                    
                    intersection = (pred_c & true_c).sum()
                    union = (pred_c | true_c).sum()
                    
                    iou_c = (intersection.float() + 1e-6) / (union.float() + 1e-6)
                    batch_iou_list.append(iou_c.item())
                
                avg_batch_iou = np.mean(batch_iou_list)
                total_val_iou += avg_batch_iou
        
        epoch_val_iou = total_val_iou / len(val_loader)
        print(f"  VALID | Epoch Mean IoU (mIoU): {epoch_val_iou:.4f}")

        
        scheduler.step(epoch_val_iou)
        
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            print(f"    --> New best mIoU: {best_val_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"    No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {epoch+1} epochs. ---")
            torch.save(model.state_dict(), SAVE_PATH_IoU)
            break
    
    print("--- Training Finished ---")
    torch.save(model.state_dict(), SAVE_PATH)

