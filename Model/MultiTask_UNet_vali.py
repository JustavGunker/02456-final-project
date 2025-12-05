import os
import numpy as np
import glob
import PIL.Image as Image
import random
import tifffile as tiff
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from Dataloader import ImageMaskDataset, unlabeledData
from MultiTask_Unet import SemiSupervisedUNet
from itertools import cycle
torch.manual_seed(28)
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

datasetSupervised = ImageMaskDataset(
    PROJECT_ROOT / "Original Images",
    PROJECT_ROOT / "Original Masks",
    resize=(512, 512)
)

datasetUnlabled = unlabeledData(
    PROJECT_ROOT / "unlabled",
    resize=(512, 512)
)

train_size = int(0.8 * len(datasetSupervised))
val_size = int(0.1 * len(datasetSupervised))
test_size = len(datasetSupervised) - train_size - val_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(
    datasetSupervised, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False)
unlabled_loader = DataLoader(datasetUnlabled, batch_size=6, shuffle=True)


def train_semi_supervised_best_model(
    model, 
    lab_loader, 
    unlab_loader, 
    val_loader,
    alpha,
    beta,
    optimizer, 
    num_epochs,
    lambda_recon=1,
    device="cuda",
    save_path="best_model.pth",
    metrics_path="training_metrics.pth"
):
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    model = model.to(device)
    best_dice = 0.0
    best_IoU = 0.0

    # Dictionary to store metrics
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Cycle the labeled loader so it's reused if shorter than unlabeled
        lab_iter = cycle(lab_loader)

        for unlabeled_batch in unlab_loader:

            # ---- Unlabeled batch ----
            imgs_u = unlabeled_batch.to(device)
            recon_u, _ = model(imgs_u, do_segmentation=False)
            loss_u = mse_loss(recon_u, imgs_u)

            # ---- Labeled batch (cycled) ----
            imgs_l, masks = next(lab_iter)
            imgs_l, masks = imgs_l.to(device), masks.to(device)

            recon_l, seg_l = model(imgs_l)

            loss_seg = alpha * dice_loss(seg_l, masks) + beta * bce_loss(seg_l, masks)
            loss_recon = mse_loss(recon_l, imgs_l)
            loss_l = loss_seg + lambda_recon * loss_recon

            # ----- Total semi-supervised loss -----
            loss = loss_u + loss_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(unlab_loader)
        metrics["train_loss"].append(avg_train_loss)

        # ----- Validation every epoch -----
        val_dice, val_iou, val_loss = validate(model, val_loader, device)
        metrics["val_loss"].append(val_loss)
        metrics["val_dice"].append(val_dice)
        metrics["val_iou"].append(val_iou)

        # Save model if Dice improved
        if val_dice > best_dice:
            best_dice = val_dice
            best_IoU = val_iou
            torch.save(model.state_dict(), save_path)

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{num_epochs} — "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Dice: {val_dice:.4f}, IoU: {val_iou:.4f}"
            )

    print(f"Training finished. Best model saved to {save_path} with Dice: {best_dice:.4f} and IoU: {best_IoU:.4f}")

    # Save metrics to a file for later use
    torch.save(metrics, metrics_path)
    print(f"Training metrics saved to {metrics_path}")

    return metrics


def validate(model, val_loader, device):
    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()
    losses = []
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            _, seg_pred = model(imgs)

            loss = bce_loss(seg_pred, masks)
            losses.append(loss.item())

            dice_scores.append(dice_coefficient(seg_pred, masks).item())
            iou_scores.append(iou_score(seg_pred, masks).item())

    return (
        sum(dice_scores)/len(dice_scores),
        sum(iou_scores)/len(iou_scores),
        sum(losses)/len(losses)
    )




def dice_coefficient(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

def iou_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()



def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
num_classes = 1
model = SemiSupervisedUNet(num_classes)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=0.05)
num_epochs = 400
metrics = train_semi_supervised_best_model(
    model=model, 
    lab_loader=train_loader, 
    unlab_loader=unlabled_loader, 
    val_loader=val_loader,
    alpha=0.1,
    beta=0.9, 
    optimizer=optimizer,
    num_epochs=num_epochs)
