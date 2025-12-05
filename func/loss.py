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

    
class ComboLoss(nn.Module):
    """
    Implements a simple, non-exponential Combo Loss.
    
    L_Combo = α * L_Dice + β * L_WCE
    """
    
    def __init__(self, dice_loss_fn, wce_loss_fn, alpha=0.5, beta=0.5):
        """
        Args:
            dice_loss_fn: An instance of your DiceLoss.
            wce_loss_fn: An instance of your CrossEntropyLoss.
            alpha (float): Weight for the Dice component.
            beta (float): Weight for the WCE component.
        """
        super().__init__()
        self.dice_loss_fn = dice_loss_fn
        self.wce_loss_fn = wce_loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        # inputs are logits (B, C, D, H, W)
        # targets are indices (B, D, H, W)
        
        # Calculate DiceLoss (0=good, 1=bad)
        loss_dice = self.dice_loss_fn(inputs, targets) 
        
        # Calculate CrossEntropyLoss (0=good, inf=bad)
        loss_wce = self.wce_loss_fn(inputs, targets.squeeze(1).long())

        # Combine them with their weights
        total_loss = self.alpha * loss_dice + self.beta * loss_wce
        
        return total_loss

class DiceLoss(nn.Module):
    """
    Implements a multi-class Dice loss.
    
    Expects:
    - y_pred: Raw, unnormalized logits from the model
              Shape: [B, C, H, W] (or [B, C, D, H, W] for 3D)
    - y_true: Ground truth labels (integers)
              Shape: [B, 1, H, W] (or [B, 1, D, H, W] for 3D)
    """
    def __init__(self, num_classes, smooth=1e-6, include_background=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.start_channel = 0 if include_background else 1

    def forward(self, y_pred, y_true):
        # y_pred_probs shape: [B, C, H, W]
        y_pred_probs = F.softmax(y_pred, dim=1)

        # y_true_one_hot shape: [B, C, H, W]
        y_true_one_hot = F.one_hot(y_true.squeeze(1).long(), num_classes=self.num_classes)
        # For 3D (B, D, H, W, C) -> (B, C, D, H, W)
        dims = list(range(len(y_true_one_hot.shape)))
        dims.insert(1, dims.pop()) # Move last dim (C) to second dim
        y_true_one_hot = y_true_one_hot.permute(*dims).float()
        
        # Shape: [B, C, -1]
        y_pred_flat = y_pred_probs.view(y_pred_probs.shape[0], self.num_classes, -1)
        y_true_flat = y_true_one_hot.view(y_true_one_hot.shape[0], self.num_classes, -1)

        # 4. Calculate intersection and union per class (over the batch)
        intersection = (y_pred_flat * y_true_flat).sum(dim=2)
        union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)
        
        # 5. Calculate Dice score per class
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 6. Average Dice score across the specified classes (e.g., ignoring background)
        dice_loss = 1 - dice_per_class[:, self.start_channel:].mean()
        
        return dice_loss
    

class FocalLoss(nn.Module):
    """
    Explicit 3D Focal Loss implementation.
    L = -weight * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter.
            weight (torch.Tensor, optional): Class weights (C,).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, D, H, W) - Unnormalized logits
            targets: (B, D, H, W) or (B, 1, D, H, W) - Long indices
        """
        # 1. Handle Target Shape (B, 1, D, H, W) -> (B, D, H, W)
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        targets = targets.long()

        # 2. Compute Log Softmax (numerical stability)
        # log_probs: (B, C, D, H, W)
        log_probs = F.log_softmax(inputs, dim=1)

        # 3. Gather Log Probabilities of the True Class
        # We need to unsqueeze targets to (B, 1, D, H, W) to use gather on dim 1
        targets_unsqueezed = targets.unsqueeze(1)
        # log_pt: (B, 1, D, H, W)
        log_pt = log_probs.gather(1, targets_unsqueezed)
        
        # Squeeze back to (B, D, H, W)
        log_pt = log_pt.squeeze(1)
        pt = log_pt.exp()

        # 4. Compute Focal Term
        focal_term = (1 - pt) ** self.gamma

        # 5. Compute Basic Loss
        loss = -focal_term * log_pt

        # 6. Apply Class Weights (if provided)
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            
            # Broadcast weights: weight[targets] creates a tensor of shape (B, D, H, W)
            # containing the weight corresponding to the class at each voxel
            loss = loss * self.weight[targets]

        # 7. Reduction
        if self.reduction == 'mean':
            if self.weight is not None:
                # Weighted average: sum(loss) / sum(weights for valid pixels)
                # This prevents loss explosion when using high class weights
                return loss.sum() / self.weight[targets].sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
## KL Divergence Loss
def kld_loss(mu, log_var):
    """
    Computes the KL Divergence loss between the learned latent distribution
    and the standard normal distribution.
    Args:
        mu (torch.Tensor): Mean of the latent distribution.
        log_var (torch.Tensor): Log variance of the latent distribution.
    Returns:
        torch.Tensor: KL Divergence loss.
    """
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

class KLAnnealing:
    def __init__(self, start_epoch, end_epoch, start_beta=0.0, end_beta=1.0):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_beta = start_beta
        self.end_beta = end_beta
    
    def get_beta(self, epoch):
        if epoch < self.start_epoch:
            return self.start_beta
        elif epoch >= self.end_epoch:
            return self.end_beta
        else:
            # Linear ramp-up
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.start_beta + (self.end_beta - self.start_beta) * progress
        
class TverskyLoss(nn.Module):
    """
    Implements a multi-class Tversky loss, which is a generalization
    of Dice loss. It's excellent for handling class imbalance and
    penalizing False Positives (FPs) or False Negatives (FNs)
    more heavily.

    Tversky(P, G) = (TP) / (TP + alpha*FP + beta*FN)
    Loss = 1 - Tversky(P, G)
    
    Setting alpha=0.7, beta=0.3 penalizes FPs (the "blob") more.
    Setting alpha=0.3, beta=0.7 penalizes FNs (missing) more.
    Setting alpha=0.5, beta=0.5 makes it identical to Dice.
    
    Expects:
    - y_pred: Raw, unnormalized logits from the model
              Shape: [B, C, H, W] (or [B, C, D, H, W] for 3D)
    - y_true: Ground truth labels (integers)
              Shape: [B, H, W] (or [B, D, H, W] for 3D) 
              (This loss expects the channel dim to be squeezed)
    """
    def __init__(self, num_classes, alpha=0.7, beta=0.3, smooth=1e-6, include_background=False):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.start_channel = 0 if include_background else 1

    def forward(self, y_pred, y_true):
        # 1. Apply Softmax to model output to get probabilities
        y_pred_probs = F.softmax(y_pred, dim=1)

        # 2. Convert ground truth labels to one-hot format
        # y_true shape: [B, D, H, W]
        y_true_one_hot = F.one_hot(y_true.long(), num_classes=self.num_classes)
        # Permute to match [B, C, D, H, W]
        dims = list(range(len(y_true_one_hot.shape)))
        dims.insert(1, dims.pop()) 
        y_true_one_hot = y_true_one_hot.permute(*dims).float()
        
        # 3. Flatten tensors but keep batch and class dims
        y_pred_flat = y_pred_probs.view(y_pred_probs.shape[0], self.num_classes, -1)
        y_true_flat = y_true_one_hot.view(y_true_one_hot.shape[0], self.num_classes, -1)

        # 4. Calculate components
        true_pos = (y_pred_flat * y_true_flat).sum(dim=2)
        false_pos = (y_pred_flat * (1 - y_true_flat)).sum(dim=2)
        false_neg = ((1 - y_pred_flat) * y_true_flat).sum(dim=2)
        
        # 5. Calculate Tversky score per class
        tversky_index = (true_pos + self.smooth) / \
            (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        # 6. Average score across the specified classes (e.g., ignoring background)
        tversky_loss = 1 - tversky_index[:, self.start_channel:].mean()
        
        return tversky_loss