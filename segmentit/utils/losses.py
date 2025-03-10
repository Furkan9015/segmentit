"""Loss functions for segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard examples.
    
    Reference:
        Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection."
        IEEE ICCV.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive examples
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (B, ...)
            targets: Target values (B, ...)
            
        Returns:
            Loss value
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal weighting
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Apply focal weighting
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks.
    
    Dice Loss optimizes the overlap between predicted and ground truth segments.
    It's especially useful for imbalanced segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (B, ...)
            targets: Target values (B, ...)
            
        Returns:
            Loss value
        """
        # Flatten inputs and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        if self.reduction == 'mean' or self.reduction == 'sum':
            return 1.0 - dice
        else:  # 'none'
            return 1.0 - dice


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss.
    
    This loss applies different weights to positive and negative examples,
    which is useful for imbalanced segmentation tasks.
    """
    
    def __init__(self, pos_weight: float = 5.0, reduction: str = 'mean'):
        """Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive examples
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (B, ...)
            targets: Target values (B, ...)
            
        Returns:
            Loss value
        """
        # Create weights tensor based on targets
        weights = targets * self.pos_weight + (1 - targets)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply weights
        weighted_loss = weights * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss


class CombinedSegmentationLoss(nn.Module):
    """Combined loss for segmentation tasks.
    
    This loss combines BCE and Dice losses for better segmentation results.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 5.0,
        smooth: float = 1.0,
    ):
        """Initialize Combined Segmentation Loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            pos_weight: Weight for positive examples in BCE loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Create loss functions
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            inputs: Predicted probabilities (B, ...)
            targets: Target values (B, ...)
            
        Returns:
            Combined loss value
        """
        # Calculate individual losses
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # Combine losses
        return self.bce_weight * bce + self.dice_weight * dice