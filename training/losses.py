"""
Loss Functions for Universal Medical Image Segmentation

This module implements various loss functions commonly used in medical image
segmentation, including Dice loss, Focal loss, and their variants optimized
for multi-class and binary segmentation scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    
    Implements the Dice coefficient as a loss function with adaptive weighting
    based on false positive and false negative ratios.
    """

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets):
        """
        Compute Dice loss.
        
        Args:
            preds: Predicted logits [N, C, ...]
            targets: Ground truth labels [N, 1, ...]
            
        Returns:
            Dice loss value
        """
        N = preds.size(0)
        C = preds.size(1)
        
        P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.) 

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P 
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        
        # Adaptive alpha based on FP/FN ratio
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8) 
        self.beta = 1 - self.alpha
        
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        if self.size_average:
            loss /= C

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses learning on hard examples by down-weighting easy examples
    using a modulating factor based on prediction confidence.
    """
    
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets):
        """
        Compute Focal loss.
        
        Args:
            preds: Predicted logits [N, C, ...]
            targets: Ground truth labels [N, 1, ...]
            
        Returns:
            Focal loss value
        """
        N = preds.size(0)
        C = preds.size(1)

        targets = targets.unsqueeze(1)
        P = F.softmax(preds, dim=1)
        log_P = F.log_softmax(preds, dim=1)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.)
        
        if targets.size(1) == 1:
            targets = targets.squeeze(1)
        alpha = self.alpha[targets.data].to(preds.device)

        probs = (P * class_mask).sum(1)
        log_probs = (log_P * class_mask).sum(1)
        
        # Apply focal weighting: (1-p)^gamma
        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss for multi-label segmentation.
    
    Computes Dice loss for each class independently, treating each as
    a binary segmentation problem. Includes class weighting to handle
    missing or padding classes.
    """

    def __init__(self, alpha=0.5, beta=0.5, class_num=72):
        super(BinaryDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # Initialize class weights (set last class weight to 0 for padding)
        weights = torch.ones(class_num, dtype=torch.float)
        weights[-1] = 0  # Padding class has zero weight
        self.register_buffer('weights', weights)

    def forward(self, preds, class_mask, tgt_idx):
        """
        Compute Binary Dice loss.
        
        Args:
            preds: Predicted logits [N, C, ...]
            class_mask: Binary ground truth masks [N, C, ...]
            tgt_idx: Target class indices [N, C]
            
        Returns:
            Weighted Binary Dice loss
        """
        N = preds.size(0)
        C = preds.size(1)

        P = F.sigmoid(preds)
        smooth = 1e-6  # Smoothing factor for numerical stability

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P 
        class_mask_ = ones - class_mask

        # Calculate True Positives, False Positives, False Negatives
        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        # Adaptive alpha based on FP/FN ratio
        self.alpha = FP.reshape(N*C, -1).sum(dim=(1)) / ((FP.reshape(N*C, -1).sum(dim=(1)) + FN.reshape(N*C, -1).sum(dim=(1))) + smooth)
        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8) 
        self.beta = 1 - self.alpha
        
        # Calculate Dice coefficient
        num = torch.sum(TP.reshape(N*C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.reshape(N*C, -1), dim=(1)).float() + self.beta * torch.sum(FN.reshape(N*C, -1), dim=(1)).float()
        
        dice = num / (den + smooth)
        loss = 1 - dice
        
        # Apply class weights based on target indices
        tgt_idx = tgt_idx.reshape(N*C)
        loss = loss * self.weights[tgt_idx]

        # Normalize by sum of weights to handle varying number of valid classes
        loss = loss.sum()
        loss /= self.weights[tgt_idx].sum()

        return loss


class BinaryCrossEntropyLoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Applies binary cross-entropy loss with class weighting to handle
    multi-label segmentation scenarios and class imbalance.
    """
    
    def __init__(self, class_num=72): 
        super(BinaryCrossEntropyLoss, self).__init__()

        # Initialize class weights (set last class weight to 0 for padding)
        weights = torch.ones(class_num, dtype=torch.float)
        weights[-1] = 0  # Padding class has zero weight
        self.register_buffer('weights', weights)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, class_mask, tgt_idx):
        """
        Compute weighted Binary Cross-Entropy loss.
        
        Args:
            preds: Predicted logits [N, C, ...]
            class_mask: Binary ground truth masks [N, C, ...]
            tgt_idx: Target class indices [N, C]
            
        Returns:
            Weighted Binary Cross-Entropy loss
        """
        N, C = preds.size(0), preds.size(1)
        
        # Compute BCE loss for each pixel
        loss = self.loss(preds, class_mask.float())

        # Average over spatial dimensions
        loss = torch.mean(loss.reshape(N*C, -1), dim=1)
        
        # Apply class weights based on target indices
        tgt_idx = tgt_idx.reshape(N*C)
        loss = loss * self.weights[tgt_idx]

        # Normalize by sum of weights
        loss = loss.sum()
        loss /= self.weights[tgt_idx].sum()

        return loss


if __name__ == '__main__':
    """Test loss functions with sample data."""
    
    # Test multi-class losses
    DL = DiceLoss()
    FL = FocalLoss(10)
    
    # 2D test
    pred = torch.randn(2, 10, 128, 128)
    target = torch.zeros((2, 1, 128, 128)).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)
    print('2D Loss - Dice:', dl_loss.item(), 'Focal:', fl_loss.item())

    # 3D test
    pred = torch.randn(2, 10, 64, 128, 128)
    target = torch.zeros(2, 1, 64, 128, 128).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)
    print('3D Loss - Dice:', dl_loss.item(), 'Focal:', fl_loss.item())

    # Test binary losses
    BDL = BinaryDiceLoss()
    BCEL = BinaryCrossEntropyLoss()
    
    pred = torch.randn(2, 5, 32, 64, 64)
    mask = torch.randint(0, 2, (2, 5, 32, 64, 64)).float()
    tgt_idx = torch.randint(0, 72, (2, 5))
    
    bdl_loss = BDL(pred, mask, tgt_idx)
    bce_loss = BCEL(pred, mask, tgt_idx)
    print('Binary Loss - Dice:', bdl_loss.item(), 'BCE:', bce_loss.item())
