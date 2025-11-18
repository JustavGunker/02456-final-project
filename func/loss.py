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






class ExpLogComboLoss(nn.Module):
    """
    Implements the Exponential Logarithmic Loss from the provided paper.
    
    LExp-Log = α * LExp-Log-Dice + β * LExp-Log-WCE
    
    Where:
    LExp-Log-Dice = (-log(Dice_Score))^γ_Dice
    LExp-Log-WCE = (WCE_Loss)^γ_WCE 

    Inspired by:
    https://arxiv.org/html/2312.05391v1
    """
    
    def __init__(self, dice_loss_fn, wce_loss_fn, alpha=0.5, beta=0.5, 
                 gamma_dice=1.0, gamma_wce=1.0, epsilon=1e-6):
        """
        Args:
            dice_loss_fn: An instance of your DiceLoss.
            wce_loss_fn: An instance of your CrossEntropyLoss.
            alpha (float): Weight for the Dice component.
            beta (float): Weight for the WCE component.
            gamma_dice (float): "Focus" parameter for Dice. >1 focuses on hard examples.
            gamma_wce (float): "Focus" parameter for WCE. >1 focuses on hard examples.
            epsilon (float): Small value for numerical stability (to avoid log(0)).
        """
        super().__init__()
        self.dice_loss_fn = dice_loss_fn
        self.wce_loss_fn = wce_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.gamma_dice = gamma_dice
        self.gamma_wce = gamma_wce
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # inputs are logits (B, C, D, H, W)
        # targets are indices (B, D, H, W)
        
        # DiceLoss (0=good, 1=bad)
        loss_dice = self.dice_loss_fn(inputs, targets) 
        
        # CrossEntropyLoss (0=good, inf=bad)
        loss_wce = self.wce_loss_fn(inputs, targets)  


        dice_score = 1.0 - loss_dice
        
        dice_score = torch.clamp(dice_score, self.epsilon, 1.0)
        
        exp_log_dice = (-torch.log(dice_score)) ** self.gamma_dice
        
        exp_log_wce = (loss_wce) ** self.gamma_wce

        total_loss = self.alpha * exp_log_dice + self.beta * exp_log_wce
        
        return total_loss
    
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
    def __init__(self, num_classes, smooth=1e-6, include_background=False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        # By default, we calculate loss for all classes, including background.
        # If you want to ignore background, set include_background=False
        self.start_channel = 0 if include_background else 1

    def forward(self, y_pred, y_true):
        # 1. Apply Softmax to model output to get probabilities
        # y_pred_probs shape: [B, C, H, W]
        y_pred_probs = F.softmax(y_pred, dim=1)

        # 2. Convert ground truth labels to one-hot format
        # y_true_one_hot shape: [B, C, H, W]
        y_true_one_hot = F.one_hot(y_true.squeeze(1).long(), num_classes=self.num_classes)
        # Permute to match [B, C, H, W] or [B, C, D, H, W]
        # For 2D (B, H, W, C) -> (B, C, H, W)
        # For 3D (B, D, H, W, C) -> (B, C, D, H, W)
        dims = list(range(len(y_true_one_hot.shape)))
        dims.insert(1, dims.pop()) # Move last dim (C) to second dim
        y_true_one_hot = y_true_one_hot.permute(*dims).float()
        
        # 3. Flatten tensors but keep batch and class dims
        # Shape: [B, C, -1]
        y_pred_flat = y_pred_probs.view(y_pred_probs.shape[0], self.num_classes, -1)
        y_true_flat = y_true_one_hot.view(y_true_one_hot.shape[0], self.num_classes, -1)

        # 4. Calculate intersection and union per class (over the batch)
        # Sum over the last dim (pixels)
        intersection = (y_pred_flat * y_true_flat).sum(dim=2)
        union = y_pred_flat.sum(dim=2) + y_true_flat.sum(dim=2)
        
        # 5. Calculate Dice score per class
        # Add smooth to avoid 0/0
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 6. Average Dice score across the specified classes (e.g., ignoring background)
        # We take the mean over the classes (dim=1) and then over the batch (dim=0)
        dice_loss = 1 - dice_per_class[:, self.start_channel:].mean()
        
        return dice_loss
    

class FocalLoss(nn.Module):
    """
    L_Focal = (1 - pt)^gamma * L_CE
    
    Where pt is the probability of the correct class.
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        Args:
            gamma (float): The "focusing" parameter. Higher values
                           focus more on hard examples.
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model logits (B, C, D, H, W)
            targets (torch.Tensor): Ground truth labels (B, 1, D, H, W)
        """
        # Ensure targets are the correct shape and type
        # (B, 1, D, H, W) -> (B, D, H, W)
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        # Calculate the standard cross entropy loss, but *without* reduction
        # This is L_CE = -log(pt)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the correct class (pt)
        # pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Calculate the final focal loss
        # L_Focal = (1 - pt)^gamma * L_CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply the specified reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
class BoundaryLoss(nn.Module):
    """
    Implements the Boundary Loss (L_B) from Eq. 23.
    L_B = integral( phi_G(q) * s(q) )
    
    phi_G(q) is the pre-computed distance map.
    s(q) is the model's softmax probability for the foreground class.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, dist_map):
        """
        Args:
            inputs (torch.Tensor): Model logits (B, C, D, H, W)
            dist_map (torch.Tensor): Pre-computed dist map (B, 1, D, H, W)
        """
        # Convert logits to probabilities
        s = F.softmax(inputs, dim=1)
        
        # We only care about the foreground class (class 1)
        # You may need to change '1' if your liver is a different class index
        s_foreground = s[:, 1, ...].unsqueeze(1) # -> (B, 1, D, H, W)
        
        # L_B = element-wise multiplication and then sum
        # This is the integral( phi_G(q) * s(q) )
        loss = torch.sum(s_foreground * dist_map)
        
        # Normalize by batch size
        return loss / inputs.size(0)

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