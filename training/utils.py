"""
Training Utilities for Universal Medical Image Segmentation

This module provides utility functions for training including:
- Optimizer setup with LoRA parameter handling
- Learning rate scheduling with warmup
- EMA (Exponential Moving Average) updates
- Distributed training utilities
- Logging functions for tensorboard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


def get_optimizer(args, net):
    """
    Create optimizer with separate learning rates for LoRA and regular parameters.
    
    Args:
        args: Configuration arguments containing optimizer settings
        net: Model network
        
    Returns:
        Configured optimizer with parameter groups
    """
    # Separate LoRA and non-LoRA parameters
    lora_params = []
    other_params = []
    
    # Identify LoRA parameters from the text encoder
    for name, param in net.named_parameters():
        if hasattr(net, 'text_encoder') and 'text_encoder' in name:
            # In PEFT models, LoRA parameters are trainable while base model params are frozen
            if param.requires_grad:
                lora_params.append(param)
            continue
        other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': other_params, 'lr': args.base_lr},  # Regular learning rate for non-LoRA params
        {'params': lora_params, 'lr': args.base_lr * 0.1}  # Lower learning rate for LoRA params
    ]
    
    # Select optimizer based on configuration
    if args.optimizer == 'sgd':
        return optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(param_groups, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(param_groups, betas=args.betas, weight_decay=args.weight_decay, eps=1e-5)
    else:
        # Default to AdamW if optimizer not specified
        return optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-5)


def log_evaluation_result(writer, dice_list, name, epoch, args, prefix):
    """
    Log evaluation results to tensorboard.
    
    Args:
        writer: Tensorboard writer
        dice_list: Array of dice scores
        name: Dataset name
        epoch: Current epoch
        args: Configuration arguments
        prefix: Logging prefix ('val' or 'test')
    """
    if writer is None:
        return
        
    C = len(dice_list) 
    writer.add_scalar(f"{prefix}_{name}_Dice/AVG", dice_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar(f"{prefix}_{name}_Dice/Dice{idx+1}", dice_list[idx], epoch+1)


def log_overall_result(writer, dice_list, epoch, args, prefix):
    """
    Log overall evaluation results across all datasets.
    
    Args:
        writer: Tensorboard writer
        dice_list: Array of dice scores
        epoch: Current epoch
        args: Configuration arguments
        prefix: Logging prefix ('val' or 'test')
    """
    if writer is None:
        return
        
    writer.add_scalar(f"{prefix}_All/Dice_AVG", dice_list.mean(), epoch+1)


def unwrap_model_checkpoint(net, ema_net, args):
    """
    Extract state dictionaries from models for checkpointing.
    
    Handles distributed training and torch.compile wrappers.
    
    Args:
        net: Main training model
        ema_net: EMA model (optional)
        args: Configuration arguments
        
    Returns:
        net_state_dict: State dict of main model
        ema_net_state_dict: State dict of EMA model (or None)
    """
    # Extract state dict handling distributed and compiled models
    net_state_dict = net.module if args.distributed else net 
    net_state_dict = net_state_dict._orig_mod.state_dict() if args.torch_compile else net_state_dict.state_dict()
    
    if args.ema and ema_net is not None:
        if args.distributed:
            ema_net_state_dict = ema_net.module.state_dict()
        else:   
            ema_net_state_dict = ema_net.state_dict()
    else:    
        ema_net_state_dict = None 

    return net_state_dict, ema_net_state_dict


def get_lr_scheduler(optimizer, init_lr, epoch, warmup_epoch, max_epoch, scheduler_type='exp'):
    """
    Learning rate scheduler with warmup support.
    
    Args:
        optimizer: Optimizer whose learning rate is being scheduled
        init_lr: Initial learning rate
        epoch: Current epoch number
        warmup_epoch: Number of warmup epochs
        max_epoch: Total number of epochs
        scheduler_type: Type of scheduler ('exp', 'cosine', or 'cosine_restart')
        
    Returns:
        Current learning rate
    """
    if scheduler_type == 'cosine_restart':
        restart_epoch = 200  # Restart point
        warmup_length = 10   # Warmup length after restart
        
        if epoch < restart_epoch:
            # First cycle - use original cosine schedule
            if epoch < warmup_epoch:
                lr = init_lr * ((epoch + 1) / warmup_epoch)
            else:
                lr = init_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epoch) / (restart_epoch - warmup_epoch)))
        else:
            # After restart
            epochs_since_restart = epoch - restart_epoch
            
            if epochs_since_restart < warmup_length:
                # Gradual warmup after restart
                start_lr = init_lr * 0.01  # Start from 1% of base_lr
                lr = start_lr + (init_lr - start_lr) * (epochs_since_restart / warmup_length)
            else:
                # Cosine decay after warmup
                effective_epoch = epochs_since_restart - warmup_length
                remaining_epochs = max_epoch - restart_epoch - warmup_length
                lr = init_lr * 0.5 * (1 + math.cos(math.pi * effective_epoch / remaining_epochs))
                lr = max(lr, init_lr * 0.01)  # Ensure lr doesn't go below 1% of base_lr
        
        # Apply learning rates to all parameter groups
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:  # Base network parameters
                param_group['lr'] = lr
            else:  # LoRA parameters
                param_group['lr'] = lr * 0.1  # LoRA params get 0.1x the base lr
        
        return lr

    elif scheduler_type == 'cosine':
        # Cosine annealing scheduler
        if epoch < warmup_epoch:
            lr = init_lr * ((epoch + 1) / warmup_epoch)
        else:
            lr = init_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epoch) / (max_epoch - warmup_epoch)))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    elif scheduler_type == 'exp':
        # Exponential scheduler with warmup
        if epoch >= 0 and epoch <= warmup_epoch:
            lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
            if epoch == warmup_epoch:
                lr = init_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr

        # Exponential decay after warmup
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):
    """
    Exponential learning rate scheduler with warmup (legacy function).
    
    This function maintains backward compatibility.
    """
    return get_lr_scheduler(optimizer, init_lr, epoch, warmup_epoch, max_epoch, scheduler_type='exp')


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Update Exponential Moving Average (EMA) model parameters.
    
    Args:
        model: Main training model
        ema_model: EMA model to update
        alpha: EMA decay factor
        global_step: Current global training step
    """
    # Adjust alpha based on training progress
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    
    # Update EMA parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # Update EMA buffers
    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(m_buffer)


@torch.no_grad()
def concat_all_gather(tensor):
    """ 
    Performs all_gather operation on the provided tensor for distributed training.
    
    Args:
        tensor: Tensor to gather across all processes
        
    Returns:
        Concatenated tensor from all processes
        
    Warning: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
    """ 
    Remove padded samples added by DistributedSampler.
    
    The DistributedSampler pads samples to evenly distribute them across GPUs.
    These padded samples need to be removed for correct evaluation.
    
    Args:
        tensor: Tensor with padded samples
        ranks: Number of processes with padded samples
        
    Returns:
        Tensor with padded samples removed
    """
    if ranks == 0:
        return tensor

    world_size = dist.get_world_size()
    single_length = len(tensor) // world_size
    output = []

    for rank in range(world_size):
        sub_tensor = tensor[rank * single_length : (rank+1) * single_length]
        if rank >= ranks:
            output.append(sub_tensor[:-1])
        else:
            output.append(sub_tensor)

    output = torch.cat(output)
    return output
