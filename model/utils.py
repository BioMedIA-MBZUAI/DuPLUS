"""
Model utilities for universal medical image segmentation.

This module provides factory functions for creating different model architectures
used in universal medical image segmentation.
"""

import numpy as np
import torch
import torch.nn as nn


def get_model(args, pretrain=False, rank=0):
    """
    Factory function to create model instances based on configuration.
    
    Args:
        args: Configuration arguments containing model specifications
        pretrain: Whether to load pretrained weights
        rank: Process rank for distributed training
        
    Returns:
        model: Instantiated model based on configuration
        
    Raises:
        ValueError: If invalid model name or dimension is specified
    """
    if args.dimension == '3d':
        if args.model == 'universal_medformer':
            from .dim3 import Universal_MedFormer

            return Universal_MedFormer(
                args.in_chan, 
                args.base_chan, 
                map_size=args.map_size, 
                conv_block=args.conv_block, 
                conv_num=args.conv_num, 
                trans_num=args.trans_num, 
                num_heads=args.num_heads, 
                fusion_depth=args.fusion_depth, 
                fusion_dim=args.fusion_dim, 
                fusion_heads=args.fusion_heads, 
                expansion=args.expansion, 
                attn_drop=args.attn_drop, 
                proj_drop=args.proj_drop, 
                proj_type=args.proj_type, 
                norm=args.norm, 
                act=args.act, 
                kernel_size=args.kernel_size, 
                scale=args.down_scale, 
                aux_loss=args.aux_loss, 
                tn=args.tn, 
                mn=args.mn
            )
        
        elif args.model == 'universal_resunet':
            from .dim3 import Universal_UNet
            
            net = Universal_UNet(
                in_ch=args.in_chan,
                base_ch=args.base_chan,
                scale=args.down_scale,
                kernel_size=args.kernel_size,
                block=args.block,
                num_block=args.num_block,
                norm=args.norm,
                act=args.act,
                num_prompts=args.num_prompts,
                prompt_dim=args.prompt_dim,
                use_film=args.use_film,
                rank=rank
            )
            return net

        else:
            raise ValueError('Invalid model name. Supported models: universal_medformer, universal_resunet')
    
    else:
        raise ValueError('Invalid dimension, should be \'3d\'')
