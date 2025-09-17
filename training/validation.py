"""
Validation Functions for Universal Medical Image Segmentation

This module provides validation functions for evaluating model performance
during training, including support for distributed training and different
evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from .utils import concat_all_gather
import logging
from utils import is_master
from tqdm import tqdm
from monai.metrics import DiceMetric


def validation_ddp_with_large_images(net, dataloader, args):
    """
    Validation function for distributed training with large medical images.
    
    This function handles large 3D medical images that may not fit entirely
    in GPU memory, using sliding window inference for evaluation.
    
    Args:
        net: Model to evaluate
        dataloader: Validation data loader
        args: Configuration arguments
        
    Returns:
        out_dice_mean: Mean Dice score across all validation samples
    """
    net.eval()
    dice_list = []
    
    # Import inference utility (assumed to handle sliding window inference)
    try:
        from inference.utils import get_inference
        inference = get_inference(args)
    except ImportError:
        # Fallback to simple inference if inference utils not available
        inference = simple_inference
    
    if is_master(args):
        logging.info("Evaluating on validation set")

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        
        for batch_data in iterator:
            # Unpack batch data
            images = batch_data["image"]
            labels = batch_data["label"]
            tgt_idx = batch_data["tgt"]
            mod_idx = batch_data["modality"]
            dataset_names = batch_data.get("dataset_name", ["unknown"])
            
            # Move to GPU
            images = images.cuda().float()
            labels = labels.to(torch.int8).cuda()
            tgt_idx = tgt_idx.cuda().long()
            mod_idx = mod_idx.cuda().long()
            
            # Get number of valid classes (exclude padding tokens)
            C = torch.nonzero(tgt_idx.squeeze(0) + 1).shape[0]
            
            if C == 0:  # Skip if no valid classes
                continue
                
            # Handle 2D case if needed
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            torch.cuda.empty_cache()
            
            # Perform inference
            pred = inference(net, images, mod_idx, dataset_names, args)
            
            del images
            torch.cuda.empty_cache()

            # Apply threshold for binary predictions
            pred = torch.sigmoid(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.to(torch.int8)
            
            torch.cuda.empty_cache()
            
            # Handle dimensions
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                pred = pred.squeeze(0)
                labels = labels.squeeze(0)
 
            # Truncate to valid classes
            pred = pred[:C, :, :, :]
            labels = labels[:C, :, :, :]

            torch.cuda.empty_cache()
            
            # Calculate Dice metric using MONAI
            dice_metric = DiceMetric(include_background=True, reduction="none")
            pred_batch = pred.unsqueeze(0)  # Add batch dimension
            labels_batch = labels.unsqueeze(0)  # Add batch dimension
            
            tmp_dice_list = dice_metric(pred_batch.to(torch.float32), labels_batch.to(torch.float32))
            tmp_dice_list = tmp_dice_list.squeeze(0)  # Remove batch dimension

            del pred, labels, pred_batch, labels_batch
            torch.cuda.empty_cache()

            # Handle distributed training
            if args.distributed:
                # Gather results from all GPUs
                tmp_dice_list = concat_all_gather(tmp_dice_list)
            
            # Add dice scores to list
            dice_list.extend(tmp_dice_list.cpu().numpy())

    # Handle padding samples in distributed training
    if args.distributed:
        world_size = dist.get_world_size()
        dataset_len = len(dataloader.dataset)

        padding_size = 0 if (dataset_len % world_size) == 0 else world_size - (dataset_len % world_size)
        
        # Remove padding samples if using DDP
        if padding_size > 0:
            dice_list = dice_list[:-padding_size]
    
    # Calculate mean Dice score
    if len(dice_list) > 0:
        out_dice_mean = np.mean(dice_list)
        dice_list_out = dice_list
    else:
        out_dice_mean = 0.0
        dice_list_out = []
        
    return out_dice_mean, dice_list_out


def simple_inference(net, images, mod_idx, dataset_names, args):
    """
    Simple inference function as fallback when sliding window inference is not available.
    
    Args:
        net: Model to use for inference
        images: Input images
        mod_idx: Modality indices  
        dataset_names: Dataset names for each sample
        args: Configuration arguments
        
    Returns:
        pred: Model predictions
    """
    # Simple forward pass
    pred = net(images, mod_idx, dataset_names)
    return pred


def validation(net, dataloader, args):
    """
    Simple validation function for non-distributed training.
    
    Args:
        net: Model to evaluate
        dataloader: Validation data loader
        args: Configuration arguments
        
    Returns:
        out_dice_mean: Mean Dice score across all validation samples
    """
    net.eval()
    dice_list = []
    
    # Import inference utility
    try:
        from inference.utils import get_inference
        inference = get_inference(args)
    except ImportError:
        inference = simple_inference
    
    logging.info("Evaluating")
    
    with torch.no_grad():
        iterator = tqdm(dataloader)
        
        for batch_data in iterator:
            # Unpack batch data
            images = batch_data["image"]
            labels = batch_data["label"]
            tgt_idx = batch_data["tgt"]
            mod_idx = batch_data["modality"]
            dataset_names = batch_data.get("dataset_name", ["unknown"])
            
            # Skip if no valid labels
            if labels.max() == 0:
                continue
                
            # Move to GPU
            images = images.cuda().float()
            labels = labels.to(torch.int8).cuda()
            tgt_idx = tgt_idx.cuda().long()
            mod_idx = mod_idx.cuda().long()
            
            # Get number of valid classes
            C = torch.nonzero(tgt_idx.squeeze(0) + 1).shape[0]
            
            if args.dimension == '2d':
                images = images.permute(1, 0, 2, 3)
            
            torch.cuda.empty_cache()
            pred = inference(net, images, mod_idx, dataset_names, args)

            del images

            # Apply threshold
            pred = torch.sigmoid(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = pred.to(torch.int8)
            
            torch.cuda.empty_cache()
           
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                pred = pred.squeeze(0)
                labels = labels.squeeze(0)
                    
            pred = pred[:C, :, :, :]
            labels = labels[:C, :, :, :]

            torch.cuda.empty_cache()
            
            # Calculate Dice using MONAI
            dice_metric = DiceMetric(include_background=True, reduction="none")
            pred_batch = pred.unsqueeze(0)
            labels_batch = labels.unsqueeze(0)
            
            tmp_dice_list = dice_metric(pred_batch.to(torch.float32), labels_batch.to(torch.float32))
            tmp_dice_list = tmp_dice_list.squeeze(0)

            del pred, labels, pred_batch, labels_batch

            dice_list.append(tmp_dice_list.cpu().numpy())

    # Calculate mean per class
    if len(dice_list) > 0:
        all_dice = np.array(dice_list)
        out_dice_mean = np.mean(all_dice, axis=0)
    else:
        out_dice_mean = np.array([0.0])

    return out_dice_mean
