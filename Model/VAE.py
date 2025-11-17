
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
from func.loss import DiceLoss, KLAnnealing, ComboLoss, FocalLoss, TverskyLoss
from func.Models import VAE
from func.dataloads import LiverDataset_aug, LiverUnlabeledDataset_aug

from func.loss import kld_loss
from xml.parsers.expat import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LATENT_DIM = 256
num_epochs = 300
KLD_WEIGHT = 0.0001
BATCH_SIZE = 1
INPUT_SHAPE = (128, 160, 160) # ( D, H, W)
NUM_CLASSES = 3  # Background, Segment 1, Segment 2

DATA_DIR = "./Task03_Liver_rs"
data_root_folder = Path.cwd() / DATA_DIR

if __name__ != "__main__":
    print(f"Data root folder: {data_root_folder}")

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
    # unlabeled set
    unlabeled_dataset = LiverUnlabeledDataset_aug(
        image_dir=data_root_folder, 
        subfolder="imagesUnlabelledTr",
        target_size= INPUT_SHAPE
    )
    
    #DataLoader for unlabeled data
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
    # Initialize model, loss functions, and optimizer
    model = VAE(
        in_channels=1, 
        latent_dim=LATENT_DIM, 
        NUM_CLASSES=NUM_CLASSES).to(device)
    
    #dice = DiceLoss(num_classes=NUM_CLASSES)
    Tversky = TverskyLoss(num_classes=NUM_CLASSES, alpha=0.7, beta=0.3).to(device)
    #CE   = nn.CrossEntropyLoss()
    focal = FocalLoss(gamma=2.0).to(device)

    loss_fn_recon = nn.MSELoss()
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    loss_fn_seg = ComboLoss(
        dice_loss_fn=Tversky,
        wce_loss_fn=focal,
        alpha=0.5, 
        beta=0.5   
    ).to(device)

    KLD_Annealing_start = 0
    KLD_Annealing_end   = 20
    kl_scheduler = KLAnnealing(
        start_epoch=KLD_Annealing_start,
        end_epoch=KLD_Annealing_end,
        start_beta=0.0,
        end_beta=0.05)
    
    print("--- Training the VAE on Liver Data ---")

    for epoch in range(num_epochs):        
        KLD_WEIGHT = kl_scheduler.get_beta(epoch)

        model.train()
        train_loss = 0
        epoch_seg_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        for batch_idx, ((x, y_target), (x_unlabeled)) in \
                enumerate(zip(labeled_loader, itertools.cycle(unlabeled_loader))):
            
            x = x.to(device)
            y_target = y_target.to(device)
            x_unlabeled = x_unlabeled[0].to(device)

            optimizer_model.zero_grad()

            seg_out, recon_out_labeled, z_mu, z_logvar = model(x)
            noise = torch.randn_like(x_unlabeled)*0.2 #0.1
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
            total_loss = (loss_seg * 1.5) + \
                        (loss_recon * 1.0) + \
                        (total_kld_loss * KLD_WEIGHT) # loss_recon *0.9 and loss_seg *1.0
            
            total_loss.backward()
            optimizer_model.step()
            train_loss += total_loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_recon_loss += loss_recon.item()
            epoch_kld_loss += total_kld_loss.item()

        avg_train_loss = train_loss / len(labeled_loader)
        avg_seg_loss = epoch_seg_loss / len(labeled_loader)
        avg_recon_loss = epoch_recon_loss / len(labeled_loader)
        avg_kld_loss = epoch_kld_loss / len(labeled_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Train Loss: {avg_train_loss:.4f} | KLD Beta: {KLD_WEIGHT:.4f}")
        print(f"  Seg Loss: {avg_seg_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | KLD Loss: {avg_kld_loss:.4f}")
        print(f"  Total Loss: {train_loss:.4f}")
    print("Training complete.")

cd = Path.cwd()
save_path = cd / "Trained_models" / "VAE.pth"
torch.save(model.state_dict(), save_path)
print(f"Saved trained model to {save_path}")