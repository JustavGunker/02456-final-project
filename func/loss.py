import torch
import torch.nn as nn

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