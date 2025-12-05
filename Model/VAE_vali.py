import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import os
import csv
import sys
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.loss import KLAnnealing, ComboLoss, FocalLoss, TverskyLoss, kld_loss
from func.Models import VAE
from func.dataloaders import VolumetricPatchDataset 
from func.utill import save_predictions, plot_learning_curves

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 3.0]).to(DEVICE) 
print(f"Using Class Weights: {CLASS_WEIGHTS}")

LATENT_DIM = 512
NUM_EPOCHS = 400
BATCH_SIZE = 3 
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4 # Background + 3 segments
LEARNING_RATE = 1e-4

# Weights
SEG_WEIGHT = 100.0 
RECON_WEIGHT = 1.0

OUTPUT_DIR = PROJECT_ROOT / "output_VAE_vali"
CSV_PATH =  PROJECT_ROOT / "stats" / "training_log_vae_final.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = PROJECT_ROOT / "Trained_models" / "VAE_val_best.pth"
SAVE_PATH_FINAL = PROJECT_ROOT / "Trained_models" / "VAE_val_final.pth"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

test_cols = [1,2, 33, 34]      
val_cols = [27, 28, 29, 30]
labeled_train_cols = [3,4,5,6,7,8 , 35,36,36,37,38]
unlabeled_train_cols = list(range(9, 27)) + list(range(40, 44))

40,41,42,43,44
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_mIoU'])

print(f"--- Data Splits ---")
print(f"Test (Reserved): {test_cols}")
print(f"Validation: {val_cols}")
print(f"Labeled Train: {labeled_train_cols}")
print(f"Unlabeled Train: {unlabeled_train_cols}")

if __name__ == "__main__":
    
    try:
        # 1. Labeled Training Loader
        train_dataset = VolumetricPatchDataset(
            selected_columns=labeled_train_cols,
            augment=True,
            is_labeled=True
        )
        labeled_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        # 2. Unlabeled Training Loader
        unlabeled_dataset = VolumetricPatchDataset(
            selected_columns=unlabeled_train_cols,
            augment=False,
            is_labeled=False
        )
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        # 3. Validation Loader (Labeled, No Augmentation)
        val_dataset = VolumetricPatchDataset(
            selected_columns=val_cols,
            augment=False,
            is_labeled=True
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"--- Loaders Ready ---")
        print(f"Train Batches: {len(labeled_loader)}")
        print(f"Val Batches: {len(val_loader)}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        exit()
    
    model = VAE(
        in_channels=1, 
        latent_dim=LATENT_DIM, 
        NUM_CLASSES=NUM_CLASSES
    ).to(DEVICE)
    
    # Loss Functions
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.6, beta=0.4).to(DEVICE)
    focal = FocalLoss(gamma=2.0).to(DEVICE)
    loss_fn_recon = nn.MSELoss().to(DEVICE)
    
    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=focal,
        alpha=0.4, 
        beta=0.6   
    ).to(DEVICE)

    optimizer_model = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode='max',      # MAXIMIZE IoU
        factor=0.5,      
        patience=20,     
        verbose=True
    )
    
    kl_scheduler = KLAnnealing(
        start_epoch=0,
        end_epoch=50,
        start_beta=0.0,
        end_beta=0.01)
    
    
    # --- TRAINING LOOP ---
    best_val_iou = 0.0
    patience_counter = 0
    SAVE_INTERVAL = 20

    # --- LOSS HISTORY ---
    train_loss_history = []
    val_loss_history = []
    val_iou_history = []

    print("--- Starting Training ---")

    for epoch in range(NUM_EPOCHS):        
        current_beta = kl_scheduler.get_beta(epoch)
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        epoch_seg_loss = 0
        epoch_recon_loss = 0
        
        last_x, last_y, last_recon, last_seg = None, None, None, None
        for batch_idx, ((x, y_target), x_unlabeled) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x = x.to(DEVICE)
            y_target = y_target.to(DEVICE).squeeze(1) # Remove channel dim for loss
            x_unlabeled = x_unlabeled.to(DEVICE)

            optimizer_model.zero_grad()

            # Forward Labeled
            seg_out, recon_out, z_mu, z_logvar = model(x)
            
            # Forward Unlabeled (with noise)
            noise = torch.randn_like(x_unlabeled) * 0.1 
            _, recon_unlabeled, mu_u, logvar_u = model(x_unlabeled + noise)

            # Losses
            l_seg = loss_fn_seg(seg_out, y_target)
            l_recon = loss_fn_recon(recon_out, x) + loss_fn_recon(recon_unlabeled, x_unlabeled)
            
            # KL Loss (Average over batch)
            l_kld = kld_loss(z_mu, z_logvar) + kld_loss(mu_u, logvar_u)
            l_kld = l_kld / (x.size(0) + x_unlabeled.size(0))

            total_loss = (l_seg * SEG_WEIGHT) + \
                         (l_recon * RECON_WEIGHT) + \
                         (l_kld * current_beta)
            
            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()
            epoch_seg_loss += l_seg.item()
            epoch_recon_loss += l_recon.item()

            if batch_idx == len(labeled_loader)-1:
                last_x = x.detach()
                last_y = y_target.detach()
                last_recon = recon_out.detach()
                last_seg = seg_out.detach()


        avg_train_loss = train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)
        train_loss_history.append(avg_train_loss)
        
        # --- VALIDATION ---
        model.eval()
        class_inter = np.zeros(NUM_CLASSES)
        class_union = np.zeros(NUM_CLASSES)
        loss_val = 0.0
        with torch.no_grad():
            for (x_val, y_val_seg) in val_loader:
                x_val = x_val.to(DEVICE)
                y_val_seg = y_val_seg.to(DEVICE).squeeze(1).long()
                
                val_seg_out, _,_,_ = model(x_val)
                val_preds = torch.argmax(val_seg_out, dim=1)

                loss = loss_fn_seg(val_seg_out, y_val_seg)
                loss_val += loss.item()
                for c in range(NUM_CLASSES):
                    pred_c = (val_preds == c)
                    true_c = (y_val_seg == c)

                    inter = (pred_c & true_c).sum().item()
                    union = (pred_c | true_c).sum().item()

                    class_inter[c] += inter
                    class_union[c] += union

        avg_val_loss = loss_val / len(val_loader)
        val_loss_history.append(avg_val_loss)
        class_iou =[]

        for c in range(NUM_CLASSES):
            if class_union[c] > 0:
                iou = class_inter[c]/class_union[c]
            else:
                iou = 0.0
            class_iou.append(iou)
        
        mIoU = np.mean(class_iou)

        val_iou_history.append(mIoU)

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, mIoU])
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        print(f" Avg Train Loss: {avg_train_loss:.4f} | Seg Loss: {avg_seg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Val mIoU: {mIoU:.4f} (Best: {best_val_iou:.4f})")
        print(f"  [Class IoU] C0: {class_iou[0]} | C1: {class_iou[1]:.4f} | C2: {class_iou[2]:.4f} | C3: {class_iou[3]:.4f}")
        
        # Scheduler Step
        scheduler.step(mIoU)

        # Save Best Model
        if mIoU > best_val_iou:
            best_val_iou = mIoU
            patience_count = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> New Best Model Saved!")
        else: 
            patience_count +=1
            (f"Patience count: {patience_count:.3f}")

        # Visualization
        if (epoch + 1) % SAVE_INTERVAL == 0:
            print(f"  Saving visuals for Epoch {epoch+1}...")
            save_predictions(epoch, last_x, last_y, last_recon, last_seg, OUTPUT_DIR)
    print("--- Training Finished ---")
    print("Saving model weights and curves...")
    torch.save(model.state_dict(), SAVE_PATH_FINAL)
    print(f"Best model saved {SAVE_PATH}")
    print(f"Final model saved {SAVE_PATH_FINAL}")
    print("Done.")
