"""
Utility functions for distributed training, logging, and checkpoint management.

This module provides essential utilities for:
- Distributed training coordination
- Logging configuration for multi-process training
- Model and optimizer checkpoint saving/loading
- Progress tracking and metrics monitoring
"""

import os
import logging
import torch
import torch.distributed as dist


LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logger(rank, log_path=None):
    """
    Configure logging for distributed training.
    
    Args:
        rank: Process rank (-1 for single GPU, 0 for master in distributed)
        log_path: Path to save log file (only master process writes to file)
    """
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # Only master process will print & write detailed logs
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    """
    Save training configuration to a text file for reproducibility.
    
    Args:
        args: Argument namespace containing all configuration parameters
    """
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")
            

def resume_load_optimizer_checkpoint(optimizer, args):
    """
    Resume optimizer state from checkpoint.
    
    Args:
        optimizer: PyTorch optimizer to load state into
        args: Arguments containing checkpoint load path
    """
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load, map_location=torch.device('cpu'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def resume_load_model_checkpoint(net, ema_net, args, strict=True):
    """
    Resume model state from checkpoint.
    
    Args:
        net: Primary model to load state into
        ema_net: EMA model to load state into (if using EMA)
        args: Arguments containing checkpoint load path and configuration
        strict: Whether to strictly enforce state dict key matching
    """
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    args.start_epoch = checkpoint['epoch']

    if args.ema:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'], strict=strict)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    
    Useful for tracking loss, accuracy, and other metrics during training.
    """

    def __init__(self, name, fmt=":f"):
        """
        Args:
            name: Name of the metric being tracked
            fmt: Format string for displaying values
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all tracked values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents (for weighted averaging)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Progress meter for displaying training progress with multiple metrics.
    """
    
    def __init__(self, num_batches, meters, prefix=""):
        """
        Args:
            num_batches: Total number of batches per epoch
            meters: List of AverageMeter objects to display
            prefix: String prefix for display
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Display current progress and metrics.
        
        Args:
            batch: Current batch number
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """Create format string for batch numbers."""
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" 


def is_master(args):
    """
    Check if current process is the master process in distributed training.
    
    Args:
        args: Arguments containing rank information
        
    Returns:
        bool: True if master process, False otherwise
    """
    return args.rank == 0
