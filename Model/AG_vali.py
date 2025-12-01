import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import os
import sys
import csv
import itertools
from pathlib import Path

# --- PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.utill import save_predictions, plot_learning_curves
from func.dataloaders import VolumetricPatchDataset 
from func.loss import ComboLoss, TverskyLoss, DiceLoss, FocalLoss
from func.Models import MultiTaskNet_ag as MultiTaskNet 

INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4
LATENT_DIM = 256 
BATCH_SIZE = 2
SAVE_INTERVAL = 20
NUM_EPOCHS = 400
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
ACCUM_STEPS = 4 

# Weights for Multi-Task Loss
SEG_WEIGHT = 100
RECON_WEIGHT = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CLASS_WEIGHTS = torch.tensor([0.5, 1.5, 1.0, 4.0]).to(device) 
print(f"Using Class Weights: {CLASS_WEIGHTS}")

OUTPUT_DIR = PROJECT_ROOT / "Output_AG_vali"
CSV_PATH =  PROJECT_ROOT / "stats" / "training_log_ag.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH_FINAL = PROJECT_ROOT / "Trained_models" / "AG_val_final.pth"
SAVE_PATH = PROJECT_ROOT / "Trained_models" / "AG_val_best.pth"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

test_cols = [1,2, 33, 34]      
val_cols = [27, 28, 29, 30]
labeled_train_cols = [3,4,5,6,7,8 , 35,36,36,37,38]
unlabeled_train_cols = list(range(9, 27))

with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_mIoU'])

print(f"--- Data Splits ---")
print(f"Test (Reserved): {test_cols}")
print(f"Validation: {val_cols}")
print(f"Labeled Train: {labeled_train_cols}")
print(f"Unlabeled Train: {unlabeled_train_cols}")

try:
    # 1. Labeled Dataset
    labeled_dataset = VolumetricPatchDataset(
        selected_columns=labeled_train_cols, 
        augment=True, 
        is_labeled=True
    )
    
    labeled_loader = DataLoader(
        dataset=labeled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    print("--- Labeled Loader Ready ---")

    # 2. Unlabeled Dataset
    unlabeled_dataset = VolumetricPatchDataset(
        selected_columns=unlabeled_train_cols,
        augment=False, 
        is_labeled=False
    )
    
    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )
    print("--- Unlabeled Loader Ready ---")

    # 3. Validation Loader
    val_dataset = VolumetricPatchDataset(
        selected_columns=val_cols,
        augment=False,
        is_labeled=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("--- Validation Loader Ready ---")
    print(f"Train Batches: {len(labeled_loader)}")
    print(f"Val Batches: {len(val_loader)}")

except Exception as e:
    print(f"Error creating Datasets: {e}")
    exit()

if __name__ == "__main__":
    model = MultiTaskNet(
        in_channels=1, 
        num_classes=NUM_CLASSES, 
        latent_dim=LATENT_DIM  
    ).to(device)

    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.6, beta=0.4)
    focal = FocalLoss(gamma=2.0).to(device)
    loss_fn_recon = nn.MSELoss().to(device)

    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=focal, # normally cross
        alpha=0.6, beta=0.4
    ).to(device)
    
    optimizer_model = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode='max',      
        factor=0.5,      
        patience=20,     
        verbose=True
    )

    best_val_iou = 0.0
    
    # --- LOSS HISTORY ---
    train_loss_history = []
    val_loss_history = []
    val_iou_history = []
    print("--- Starting Training ---")
    
    for epoch in range(NUM_EPOCHS):
        
        # --- TRAINe ---
        model.train() 
        epoch_train_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        
        # Visualization variables
        last_x, last_y, last_recon, last_seg = None, None, None, None

        for batch_idx, ((x, y_seg_target), (x_unlabeled)) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x = x.to(device)       
            y_seg_target = y_seg_target.to(device).squeeze(1) # Squeeze for CrossEntropy
            x_unlabeled = x_unlabeled.to(device) 
            
            optimizer_model.zero_grad()
            
            # Forward Labeled
            seg_out, recon_out_labeled = model(x)
            
            # Forward Unlabeled (with noise)
            noise = torch.randn_like(x_unlabeled) * 0.1
            x_unlabeled_noisy = x_unlabeled + noise
            _ , recon_out_unlabeled = model(x_unlabeled_noisy)
                        
            # Loss Calculation
            loss_seg = loss_fn_seg(seg_out, y_seg_target)
            loss_recon = loss_fn_recon(recon_out_labeled, x) + \
                         loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            total_loss = (loss_seg*SEG_WEIGHT) + (loss_recon*RECON_WEIGHT)
            
            total_loss.backward()
            optimizer_model.step()
            #loss_normalized = total_loss / ACCUM_STEPS
            #loss_normalized.backward()
            
            ##  Step Optimizer only every ACCUM_STEPS
            #if (batch_idx + 1) % ACCUM_STEPS == 0:
            ##    optimizer_model.step()
            #    optimizer_model.zero_grad()
            
            #total_loss.backward()
            #optimizer_model.step()
            
            epoch_train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += loss_recon.item()

            # Save last batch
            if batch_idx == len(labeled_loader) - 1:
                last_x = x.detach()
                last_y = y_seg_target.detach()
                last_recon = recon_out_labeled.detach()
                last_seg = seg_out.detach()

                # Ensure step is taken if last batch wasn't divisible
        #if len(labeled_loader) % ACCUM_STEPS != 0:
        #     optimizer_model.step()
        #     optimizer_model.zero_grad()
             
        avg_train_loss = epoch_train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)
        train_loss_history.append(avg_train_loss)
        
        # --- VALIDATION PHASE ---
        model.eval()
        #total_val_iou = 0.0
        class_inter = np.zeros(NUM_CLASSES)
        class_union = np.zeros(NUM_CLASSES)
        loss_val = 0.0
        with torch.no_grad():
            for (x_val, y_val_seg) in val_loader:
                x_val = x_val.to(device)
                y_val_seg = y_val_seg.to(device).squeeze(1).long()
                
                val_seg_out, _ = model(x_val)
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
            if class_union[c] > c:
                iou = class_inter[c]/class_union[c]
            else:
                iou = 0.0
            class_iou.append(iou)
        
        foreground = class_iou[1:]
        mIoU = np.mean(foreground)
        val_iou_history.append(mIoU)

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, mIoU])
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        print(f" Avg Train Loss: {avg_train_loss:.4f} | Seg Loss: {avg_seg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Val mIoU: {mIoU:.4f} (Best: {best_val_iou:.4f})")
        print(f" [Class IoU] C0: {class_iou[0]:.4f} C1: {class_iou[1]:.4f} | C2: {class_iou[2]:.4f} | C3: {class_iou[3]:.4f}")
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
    print("Saving model weights...")
    torch.save(model.state_dict(), SAVE_PATH_FINAL)
    print(f"Best model saved {SAVE_PATH}")
    print(f"Final model saved {SAVE_PATH_FINAL}")
    print("Done.")

    