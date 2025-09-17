"""
Universal Medical Image Segmentation Training Script

This script provides the main training pipeline for universal medical image segmentation.
It supports distributed training, mixed precision, and various model architectures.

Key Features:
- Multi-GPU distributed training support
- Automatic mixed precision (AMP) for faster training
- Exponential Moving Average (EMA) for model stabilization
- Gradient accumulation for large batch sizes
- Comprehensive validation on multiple datasets
- Tensorboard logging for training monitoring

Usage:
    python train.py --dataset universal --model MODEL_NAME --dimension 3d [options]

Example:
    python train.py --dataset universal --model universal_resunet --dimension 3d --amp --batch_size 8

Requirements:
    - PyTorch >= 1.12.0
    - MONAI >= 1.0.0
    - tensorboard
    - PyYAML
    - numpy
"""

import builtins
import logging
import os
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from training.utils import get_lr_scheduler, update_ema_variables
from training.losses import BinaryDiceLoss, BinaryCrossEntropyLoss, DiceLoss
from training.validation import validation_ddp_with_large_images as validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    log_overall_result, 
    get_optimizer,
    unwrap_model_checkpoint,
)
import yaml
import argparse
import time
import math
import sys
import warnings
import copy

import monai

from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)
warnings.filterwarnings("ignore", category=UserWarning)


def train_net(net, trainset, valset_list, testset_list, args, ema_net=None):
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        net: Primary model for training
        trainset: Training dataset
        valset_list: List of validation datasets
        testset_list: List of test datasets  
        args: Training configuration arguments
        ema_net: Exponential Moving Average model (optional)
        
    Returns:
        best_Dice: Best validation Dice scores achieved during training
    """
    ########################################################################################
    # Dataset Creation and DataLoader Setup
    
    # Weighted sampling to handle class imbalance across datasets
    samples_weight = torch.from_numpy(np.array(trainset.weight_list))
    train_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight)*10)

    # Create training dataloader with MONAI's optimized ThreadDataLoader
    trainLoader = monai.data.ThreadDataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # shuffle if no sampler
        sampler=train_sampler,
        buffer_size=2 * args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers>0)
    )

    # Create validation dataloaders for each dataset
    valLoader_list = []
    for i in range(len(args.dataset_name_list)):
        dataset_name = args.dataset_name_list[i]
        valset = valset_list[i]
        val_sampler = DistributedSampler(valset) if args.distributed else None
        valLoader = monai.data.ThreadDataLoader(
            valset,
            batch_size=1,  # batch size 1 for 3D inputs with different sizes
            shuffle=False, 
            sampler=val_sampler,
            buffer_size=2 * args.batch_size,
            pin_memory=True,
            num_workers=0
        )
        valLoader_list.append(valLoader)
    
    # Create test dataloaders for each dataset
    testLoader_list = []
    for i in range(len(args.dataset_name_list)):
        dataset_name = args.dataset_name_list[i]
        testset = testset_list[i]
        test_sampler = DistributedSampler(testset) if args.distributed else None
        testLoader = monai.data.ThreadDataLoader(
            testset,
            batch_size=1,  # batch size 1 for 3D inputs with different sizes
            shuffle=False, 
            sampler=test_sampler,
            buffer_size=2 * args.batch_size,
            pin_memory=True,
            num_workers=0
        )
        testLoader_list.append(testLoader)
    
    if is_master(args):
        logging.info(f"Created Dataset and DataLoader")

    ########################################################################################
    # Initialize training components
    
    # Setup tensorboard logging (only on master process)
    writer = SummaryWriter(os.path.join(args.exp_dir, "tensorboard")) if is_master(args) else None

    # Initialize optimizer
    optimizer = get_optimizer(args, net)
    
    # Resume optimizer state if resuming training
    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)
    
    # Define loss functions
    criterion_mod = nn.CrossEntropyLoss(ignore_index=-1).cuda(args.proc_idx)
    criterion_dice = BinaryDiceLoss(72).cuda(args.proc_idx)  # 72 max classes
    criterion_bce = BinaryCrossEntropyLoss(72).cuda(args.proc_idx)
    
    # Initialize automatic mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ########################################################################################
    # Main training loop
    
    best_Dice = np.zeros(np.array(args.dataset_classes_list).sum())
    
    for epoch in range(args.start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
            
        # Update learning rate with scheduler
        current_lr = get_lr_scheduler(optimizer, 
                                    init_lr=args.base_lr, 
                                    epoch=epoch, 
                                    warmup_epoch=args.warmup_epoch, 
                                    max_epoch=args.epochs,
                                    scheduler_type=args.scheduler)
        if is_master(args):
            logging.info(f"Current lr: {current_lr:.4e}")
       
        # Train one epoch
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, 
                   criterion_dice, criterion_bce, scaler, args)
        
        ##################################################################################
        # Evaluation and checkpoint saving
        
        # Use EMA model for evaluation if available
        net_for_eval = ema_net if args.ema else net

        # Save latest checkpoint (only on master process)
        if is_master(args):
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.exp_dir, 'latest.pth'))

        # Perform validation and testing
        if (epoch+1) % args.val_freq == 0 or (epoch+1>(args.epochs-20)):
            best_Dice = validate(valLoader_list, net_for_eval, net, ema_net, epoch, 
                               writer, optimizer, args, best_Dice, prefix='val')
            _ = validate(testLoader_list, net_for_eval, net, ema_net, epoch, 
                        writer, optimizer, args, best_Dice, prefix='test')

    return best_Dice


def validate(loader_list, net_for_eval, net, ema_net, epoch, writer, optimizer, args, best_Dice, prefix='test'):
    """
    Validation function that evaluates the model on multiple datasets.
    
    Args:
        loader_list: List of data loaders for different datasets
        net_for_eval: Model to use for evaluation (could be EMA model)
        net: Primary training model
        ema_net: EMA model
        epoch: Current epoch number
        writer: Tensorboard writer
        optimizer: Optimizer state
        args: Configuration arguments
        best_Dice: Current best Dice scores
        prefix: Evaluation prefix ('val' or 'test')
        
    Returns:
        best_Dice: Updated best Dice scores
    """
    all_dice = []
    
    # Evaluate on each dataset
    for idx in range(len(loader_list)):
        Loader = loader_list[idx]
        dataset_name = args.dataset_name_list[idx]

        # Perform validation using distributed validation function
        dice_test_mean, dice_list_test = validation(net_for_eval, Loader, args)

        if is_master(args):
            logging.info(f"{dataset_name} mean: {dice_test_mean}")
            log_evaluation_result(writer, dice_test_mean, dataset_name, epoch, args, prefix)
            
        all_dice.append(dice_test_mean)
        
    # Log overall results and save best model
    if is_master(args):
        all_dice = np.array(all_dice)
        log_overall_result(writer, all_dice, epoch, args, prefix)

        # Save best model checkpoint if validation performance improved
        if all_dice.mean() >= best_Dice.mean() and prefix == 'val':
            best_Dice = all_dice

            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.exp_dir, 'best.pth'))

        logging.info("Evaluation Done")
        logging.info(f"Dice: {all_dice.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")
    
    return best_Dice


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion_dice, criterion_bce, scaler, args):
    """
    Training function for a single epoch.
    
    Args:
        trainLoader: Training data loader
        net: Model to train
        ema_net: EMA model for stabilization
        optimizer: Optimizer
        epoch: Current epoch number
        writer: Tensorboard writer
        criterion_dice: Dice loss function
        criterion_bce: Binary cross-entropy loss function
        scaler: AMP scaler for mixed precision
        args: Configuration arguments
    """
    # Initialize progress tracking
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    epoch_loss_seg = AverageMeter("Loss_seg", ":.2f")
    
    progress = ProgressMeter(
        args.iter_per_epoch,
        [batch_time, epoch_loss_seg], 
        prefix="Epoch: [{}]".format(epoch+1),
    )
    
    net.train()

    tic = time.time()
    iter_num_per_epoch = 0
    optimizer.zero_grad()  # Zero gradients at the start of accumulation
    
    for i, batch_data in enumerate(trainLoader):
        # Unpack batch data
        img = batch_data["image"]
        label = batch_data["label"]
        emb = batch_data["embedding"]
        tgt = batch_data["tgt"]
        class_embeddings = batch_data["class_embeddings"]
        modality = batch_data["modality"]
        datasets_names = batch_data["dataset_name"]

        # Move data to GPU
        img = img.cuda(args.proc_idx, non_blocking=True).float()
        label = label.cuda(args.proc_idx, non_blocking=True).long()
        emb = emb.cuda(args.proc_idx, non_blocking=True).float()
        tgt = tgt.cuda(args.proc_idx, non_blocking=True).long()
        class_embeddings = class_embeddings.cuda(args.proc_idx, non_blocking=True).float()
        
        step = i + epoch * len(trainLoader)  # global step count
        
        loss_seg = 0
        
        # Forward pass with optional mixed precision
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                result = net(img, modality, datasets_names)
                
                # Handle multi-scale outputs from the model
                if isinstance(result, tuple) or isinstance(result, list):
                    for j in range(len(result)):
                        # Combine Dice and BCE losses with weighting
                        dice_loss = criterion_dice(result[j], label, tgt)
                        bce_loss = criterion_bce(result[j], label, tgt)
                        loss_seg += args.aux_weight[j] * (dice_loss + 0.5 * bce_loss)
                else:
                    # Single output case
                    dice_loss = criterion_dice(result, label, tgt)
                    bce_loss = criterion_bce(result, label, tgt)
                    loss_seg = dice_loss + 0.5 * bce_loss

                # Scale loss for gradient accumulation
                loss = loss_seg / args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights when accumulation is complete
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(trainLoader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
        else:
            # Standard precision training
            result = net(img, modality, datasets_names)
            
            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    dice_loss = criterion_dice(result[j], label, tgt)
                    bce_loss = criterion_bce(result[j], label, tgt)
                    loss_seg += args.aux_weight[j] * (dice_loss + 0.5 * bce_loss)
            else:
                dice_loss = criterion_dice(result, label, tgt)
                bce_loss = criterion_bce(result, label, tgt)
                loss_seg = dice_loss + 0.5 * bce_loss

            loss = loss_seg / args.gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(trainLoader):
                optimizer.step()
                optimizer.zero_grad()
        
        # Update EMA model if enabled
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        # Update progress tracking
        epoch_loss.update(loss_seg.item(), img.shape[0])
        epoch_loss_seg.update(loss_seg.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        # Print progress
        if is_master(args) and i % args.print_freq == 0:
            progress.display(i)
        
        # Handle 3D training iteration limit
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break
        
        # Log training metrics to tensorboard
        if is_master(args):
            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)
            writer.add_scalar('Train/Loss_seg', epoch_loss_seg.avg, epoch+1)


def get_parser():
    """
    Parse command line arguments and load configuration from YAML file.
    
    Returns:
        args: Parsed arguments with configuration loaded
    """
    parser = argparse.ArgumentParser(description='Universal medical image segmentation training')
    
    # Core training arguments
    parser.add_argument('--dataset', type=str, default='universal', help='dataset name')
    parser.add_argument('--model', type=str, default='universal_resunet', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision for faster training')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile to accelerate training (PyTorch 2.0+)')
    parser.add_argument('--use_film', action='store_true', help='use FiLM instead of dynamic convolution for feature modulation')

    # Training hyperparameters
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='number of gradient accumulation steps')
    
    # Checkpoint and logging
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log/', help='path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    # Hardware configuration
    parser.add_argument('--gpu', type=str, default='0,1,2,3')

    args = parser.parse_args()

    # Store command line arguments that were explicitly set
    cli_args = {key: value for key, value in vars(args).items() if value is not parser.get_default(key)}

    # Load configuration from YAML file
    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    logging.info('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Set config values but don't overwrite command line arguments
    for key, value in config.items():
        if key not in cli_args:  # Only set if not specified in command line
            setattr(args, key, value)

    # Log feature modulation method
    logging.info(f"Using {'FiLM' if args.use_film else 'Dynamic Convolution'} for feature modulation")

    return args


def init_network(args):
    """
    Initialize the network and optionally the EMA network.
    
    Args:
        args: Configuration arguments
        
    Returns:
        net: Primary training network
        ema_net: EMA network (if enabled)
    """
    net = get_model(args, pretrain=args.pretrain, rank=args.rank)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain, rank=args.rank)
        logging.info("Use EMA model for evaluation") if is_master(args) else None
    else:
        ema_net = None
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.torch_compile:
        net = torch.compile(net)

    return net, ema_net


def main_worker(proc_idx, ngpus_per_node, args, result_dict=None, trainset=None, valset=None, testset=None):
    """
    Main worker function for distributed training setup.
    
    Args:
        proc_idx: Process index (GPU index)
        ngpus_per_node: Number of GPUs per node
        args: Configuration arguments
        result_dict: Dictionary for collecting results across processes
        trainset: Training dataset
        valset: Validation dataset list
        testset: Test dataset list
    """
    # Set random seeds for reproducibility
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # Suppress printing if not master process
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    # Initialize distributed training
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        torch.cuda.set_device(args.proc_idx)
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank
        )
        
        torch.cuda.set_device(args.proc_idx)
        dist.barrier(device_ids=[args.proc_idx])

        # Adjust batch size for distributed training
        args.batch_size = int(args.batch_size / args.world_size)

    # Create experiment directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.exp_dir = os.path.join(args.cp_path, args.dataset, f"experiment_{timestamp}_{args.unique_name}")
    args.cp_dir = args.exp_dir
    
    # Synchronize processes for directory creation
    if args.distributed:
        dist.barrier()
        
        if args.rank == 0:
            os.makedirs(args.exp_dir, exist_ok=True)
            save_configure(args)
            
        dist.barrier()
        
        configure_logger(args.rank, os.path.join(args.exp_dir, "log.txt"))
        logging.getLogger().setLevel(logging.INFO)
        
        if args.rank == 0:
            logging.info(f"\nExperiment directory: {args.exp_dir}")
            logging.info(
                f"\nDataset: {args.dataset},\n"
                + f"Model: {args.model},\n"
                + f"Dimension: {args.dimension}"
            )
    else:
        os.makedirs(args.exp_dir, exist_ok=True)
        configure_logger(args.rank, os.path.join(args.exp_dir, "log.txt"))
        logging.getLogger().setLevel(logging.INFO)
        save_configure(args)
        logging.info(f"\nExperiment directory: {args.exp_dir}")
        logging.info(
            f"\nDataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension}"
        )

    # Initialize networks
    net, ema_net = init_network(args)

    # Move networks to GPU
    net.to(f"cuda")
    if args.ema:
        ema_net.to(f"cuda")

    # Setup distributed training wrappers
    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=True)
        
        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=True)
            
            for p in ema_net.parameters():
                p.requires_grad_(False)

    if is_master(args):
        logging.info(f"Created Model")
    
    # Start training
    best_Dice = train_net(net, trainset, valset, testset, args, ema_net)
    
    if is_master(args):
        logging.info(f"Training and evaluation are done")
    
    # Collect results
    if args.distributed:
        if is_master(args):
            result_dict['best_Dice'] = best_Dice
    else:
        return best_Dice


if __name__ == '__main__':
    # Parse arguments and setup
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_path = os.path.join(args.log_path, args.dataset)

    # Setup distributed training parameters
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    
    if args.world_size > 1:
        args.multiprocessing_distributed = True
    else:
        args.multiprocessing_distributed = False
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # Set multiprocessing sharing strategy
    mp.set_sharing_strategy('file_system')
    
    if args.multiprocessing_distributed:
        # Distributed training setup
        with mp.Manager() as manager:
            result_dict = manager.dict()
                
            # Create datasets
            trainset = get_dataset(args, dataset_name_list=args.dataset_name_list, mode='train') 
            valset_list = []
            for dataset_name in args.dataset_name_list:
                valset = get_dataset(args, dataset_name_list=[dataset_name], mode='val')
                valset_list.append(valset)
            testset_list = []
            for dataset_name in args.dataset_name_list:
                testset = get_dataset(args, dataset_name_list=[dataset_name], mode='test')
                testset_list.append(testset)
                
            # Launch distributed processes
            mp.spawn(main_worker, nprocs=ngpus_per_node, 
                    args=(ngpus_per_node, args, result_dict, trainset, valset_list, testset_list))
            best_Dice = result_dict['best_Dice']
    else:
        # Single GPU training
        trainset = get_dataset(args, dataset_name_list=args.dataset_name_list, mode='train')
        valset_list = []
        for dataset_name in args.dataset_name_list:
            valset = get_dataset(args, dataset_name_list=[dataset_name], mode='val')
            valset_list.append(valset)
        testset_list = []
        for dataset_name in args.dataset_name_list:
            testset = get_dataset(args, dataset_name_list=[dataset_name], mode='test')
            testset_list.append(testset)
            
        best_Dice = main_worker(0, ngpus_per_node, args, trainset=trainset, valset=valset_list, testset=testset_list)

    # Save final results
    with open(os.path.join(args.exp_dir, "results.txt"), 'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        f.write(f"Each Class Dice: {best_Dice}\n")
        f.write(f"All classes Dice Avg: {best_Dice.mean()}\n")
        f.write("\n")

    logging.info('Training done.')
    sys.exit(0)
