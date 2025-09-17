"""
Utility functions for model components.

This module provides factory functions for creating different types of
building blocks used throughout the universal segmentation architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, SingleConv
from .trans_layers import LayerNorm


def get_block(name):
    """
    Factory function to get convolutional block types.
    
    Args:
        name: String name of the block type
        
    Returns:
        Block class corresponding to the given name
    """
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
    }
    
    if name not in block_map:
        raise ValueError(f"Unknown block type: {name}. Available: {list(block_map.keys())}")
        
    return block_map[name]


def get_norm(name):
    """
    Factory function to get normalization layer types.
    
    Args:
        name: String name of the normalization type
        
    Returns:
        Normalization class corresponding to the given name
    """
    norm_map = {
        'bn': nn.BatchNorm3d,
        'in': nn.InstanceNorm3d,
        'ln': LayerNorm
    }
    
    if name not in norm_map:
        raise ValueError(f"Unknown normalization type: {name}. Available: {list(norm_map.keys())}")
        
    return norm_map[name]


def get_act(name):
    """
    Factory function to get activation function types.
    
    Args:
        name: String name of the activation function
        
    Returns:
        Activation class corresponding to the given name
    """
    act_map = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'elu': nn.ELU,
    }
    
    if name not in act_map:
        raise ValueError(f"Unknown activation type: {name}. Available: {list(act_map.keys())}")
        
    return act_map[name]
