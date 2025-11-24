import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
import sys
import itertools
from pathlib import Path
import matplotlib.pyplot as plt

# --- Custom Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from func.utill import save_predictions
from func.loss import DiceLoss, ComboLoss, TverskyLoss, FocalLoss
from func.Models import MultiTaskNet_big
from func.dataloaders import VolumetricPatchDataset

# --- CONFIGURATION ---
BLACKHOLE_PATH = os.environ.get('BLACKHOLE', '.')
INPUT_SHAPE = (128, 128, 128) 
NUM_CLASSES = 4
LATENT_DIM = 256 
BATCH_SIZE = 3 
SAVE_INTERVAL = 20
NUM_EPOCHS = 800
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
class_weight = torch.tensor([0.6, 1.0, 1.0, 3.0])

OUTPUT_DIR = PROJECT_ROOT / "output_big_vali"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = PROJECT_ROOT / "Trained_models" / "multi_big_best.pth"
SAVE_PATH_FINAL = PROJECT_ROOT / "Trained_models" / "multi_big_final.pth"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- DATA SPLITS ---
test_cols = [1,2, 33, 34]      
val_cols = [27, 28, 29, 30]
labeled_cols = [3,4,5,6,7,8 , 35,36,36,37,38]
unlabeled_cols = list(range(9, 27))

print(f"--- Data Splits ---")
print(f"Labeled Train: {labeled_cols}")
print(f"Unlabeled Train: {unlabeled_cols}")
print(f"Validation: {val_cols}")

try:
    labeled_dataset = VolumetricPatchDataset(selected_columns=labeled_cols, augment=True, is_labeled=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    unlabeled_dataset = VolumetricPatchDataset(selected_columns=unlabeled_cols, augment=False, is_labeled=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = VolumetricPatchDataset(selected_columns=val_cols, augment=False, is_labeled=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("--- Loaders Ready ---")

except Exception as e:
    print(f"Error creating Datasets: {e}")
    exit()


# --- TRAINING LOOP ---
if __name__ == "__main__":
    model = MultiTaskNet_big(in_channels=1, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(DEVICE)
    
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.6, beta=0.4).to(DEVICE)
    focal = FocalLoss(gamma=2.0, weight=class_weight).to(DEVICE)
    loss_seg_fn = ComboLoss(dice_loss_fn=Tversky, wce_loss_fn=focal).to(DEVICE)
    loss_fn_recon = nn.MSELoss().to(DEVICE)
    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for stability

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_model, mode="max", factor=0.5, patience=20, verbose=True
    )

    best_val_iou = 0.0
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 50
    
    print("--- Starting Training ---")
    
    unlabeled_iterator = itertools.cycle(unlabeled_loader)

    for epoch in range(NUM_EPOCHS):
        
        # === TRAINING ===
        model.train() 
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        
        last_x, last_y, last_recon, last_seg = None, None, None, None
        
        for batch_idx, ((x, y_seg_target), x_unlabeled) in \
                        enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x = x.to(DEVICE)
            y_seg_target = y_seg_target.to(DEVICE).squeeze(1)
            x_unlabeled = x_unlabeled.to(DEVICE) 

            optimizer_model.zero_grad()
            
            # 1. Labeled Forward
            seg_out, recon_out_labeled, _ = model(x)
            total_loss_seg = loss_seg_fn(seg_out, y_seg_target)
            loss_recon_labeled = loss_fn_recon(recon_out_labeled, x)

            # 2. Unlabeled Forward
            noise = torch.randn_like(x_unlabeled) * 0.1
            x_unlabeled_noisy = x_unlabeled + noise
            _ , recon_out_unlabeled, _ = model(x_unlabeled_noisy)
            
            loss_recon_unlabeled = loss_fn_recon(recon_out_unlabeled, x_unlabeled)
            
            total_loss_recon = loss_recon_labeled + loss_recon_unlabeled
            
            # Weighted Sum
            total_loss = (total_loss_seg * 5.0) + (total_loss_recon * 1.0)
                
            total_loss.backward()
            optimizer_model.step()
            
            train_loss += total_loss.item()
            epoch_seg_loss += total_loss_seg.item()
            epoch_recon_loss += total_loss_recon.item()
            
            if batch_idx == len(labeled_loader) - 1:
                last_x = x.detach()
                last_y = y_seg_target.detach()
                last_recon = recon_out_labeled.detach()
                last_seg = seg_out.detach()

        avg_train_loss = train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)

        # === VALIDATION ===
        model.eval()
        class_inter = np.zeros(NUM_CLASSES)
        class_union = np.zeros(NUM_CLASSES)
        
        with torch.no_grad():
            # BUG FIX 1: Use consistent variable names (vx, vy)
            for vx, vy_seg in val_loader:
                vx = vx.to(DEVICE)
                vy_seg = vy_seg.to(DEVICE).squeeze(1).long()
                
                val_seg_out, _, _ = model(vx)
                val_preds = torch.argmax(val_seg_out, dim=1)

                for c in range(NUM_CLASSES):
                    pred_c = (val_preds == c)
                    true_c = (vy_seg == c)

                    inter = (pred_c & true_c).sum().item()
                    union = (pred_c | true_c).sum().item()

                    class_inter[c] += inter
                    class_union[c] += union

        class_iou = []
        for c in range(NUM_CLASSES):
            # BUG FIX 3: Correct logic for IoU calculation
            if class_union[c] > 0:
                iou = class_inter[c] / class_union[c]
            else:
                iou = 0.0
            class_iou.append(iou)
        
        foreground = class_iou[1:]
        mIoU = np.mean(foreground)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        print(f"  Seg: {avg_seg_loss:.4f} | Recon: {avg_recon_loss:.4f}")
        print(f"  Val mIoU: {mIoU:.4f} (Best: {best_val_iou:.4f})")
        print(f"  [Class IoU] C1: {class_iou[1]:.4f} | C2: {class_iou[2]:.4f} | C3: {class_iou[3]:.4f}")
        
        scheduler.step(mIoU)

        if mIoU > best_val_iou:
            best_val_iou = mIoU
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> New Best Model Saved!")
        else: 
            patience_counter += 1
            print(f"  Patience count: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            print(f"  Saving visuals for Epoch {epoch +1}...")
            save_predictions(epoch, last_x, last_y, last_recon, last_seg, OUTPUT_DIR)

    print("--- Training Finished ---")
    torch.save(model.state_dict(), SAVE_PATH_FINAL)
    print(f"Best model saved {SAVE_PATH}")
    print(f"Final model saved {SAVE_PATH_FINAL}")
    print("Done.")