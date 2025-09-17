"""
Universal UNet Utilities

This module contains the building blocks for the Universal UNet architecture,
including input convolution, down-sampling blocks, and up-sampling blocks
with support for text-guided feature modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct
from .trans_layers import Attention, CrossAttention, LayerNorm, Mlp, PreNorm
from einops import rearrange


class inconv(nn.Module):
    """
    Initial convolution layer for UNet.
    
    Performs the first convolution operations on input images,
    typically converting from input channels to base feature channels.
    """
    
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d, act=nn.ReLU):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm, act=act)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)
        return out 


class down_block(nn.Module):
    """
    Down-sampling block for UNet encoder.
    
    Performs feature down-sampling and processing with support for
    text-guided feature modulation through prompts.
    """
    
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], 
                 down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d, use_prompts=False, 
                 use_film=False, prompt_dim=None, act=nn.ReLU):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        self.block_class = block
        self.block_list = nn.ModuleList()
        self.use_prompts = use_prompts
        self.prompt_dim = prompt_dim
        
        if pool:
            # Use max pooling for downsampling
            self.block_list.append(nn.MaxPool3d(down_scale))
            self.block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm, 
                                       prompt_dim=prompt_dim, 
                                       use_dynamic_conv=use_prompts and not use_film,
                                       use_film=use_prompts and use_film, 
                                       act=act))
        else:
            # Use strided convolution for downsampling
            self.block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm, 
                                       prompt_dim=prompt_dim,
                                       use_dynamic_conv=use_prompts and not use_film,
                                       use_film=use_prompts and use_film, 
                                       act=act))

        # Additional convolutional blocks
        for i in range(num_block-1):
            self.block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm, 
                                       use_dynamic_conv=use_prompts and not use_film,
                                       use_film=use_prompts and use_film, 
                                       prompt_dim=prompt_dim,
                                       act=act))

    def forward(self, x, prompts=None):
        """
        Forward pass with optional text prompt modulation.
        
        Args:
            x: Input feature tensor
            prompts: Text prompt embeddings for feature modulation
            
        Returns:
            x: Output feature tensor after downsampling and processing
        """
        for i, block in enumerate(self.block_list):
            if self.use_prompts and isinstance(block, self.block_class):
                if prompts is None:
                    raise ValueError("Prompts are required when use_prompts is True")
                x = block(x, prompts)
            else:
                x = block(x)
        return x


class up_block(nn.Module):
    """
    Up-sampling block for UNet decoder.
    
    Performs feature up-sampling, skip connection concatenation, and processing
    with support for text-guided feature modulation through prompts.
    """
    
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], 
                 up_scale=[2,2,2], norm=nn.BatchNorm3d, use_prompts=False, use_film=False, 
                 prompt_dim=None, act=nn.ReLU):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.block_class = block
        self.up_scale = up_scale
        self.use_prompts = use_prompts
        
        self.block_list = nn.ModuleList()
        
        # First block handles concatenated features (upsampled + skip connection)
        self.block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm, 
                                   use_dynamic_conv=use_prompts and not use_film,
                                   use_film=use_prompts and use_film,
                                   prompt_dim=prompt_dim, 
                                   act=act))
        
        # Additional blocks for further processing
        for i in range(num_block-1):
            self.block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm,
                                       use_dynamic_conv=use_prompts and not use_film,
                                       use_film=use_prompts and use_film,
                                       prompt_dim=prompt_dim,
                                       act=act))

    def forward(self, x1, x2, prompts=None):
        """
        Forward pass with skip connection and optional text prompt modulation.
        
        Args:
            x1: Input feature tensor from previous decoder layer
            x2: Skip connection feature tensor from encoder
            prompts: Text prompt embeddings for feature modulation
            
        Returns:
            x: Output feature tensor after upsampling and processing
        """
        input_dtype = x1.dtype
        # F.interpolate trilinear doesn't support bfloat16, so cast to float32 for upsampling
        x1 = F.interpolate(x1.float(), size=x2.shape[2:], mode='trilinear', align_corners=True)
        x1 = x1.to(input_dtype)
        x = torch.cat([x2, x1], dim=1)

        for i, block in enumerate(self.block_list):
            if self.use_prompts and isinstance(block, self.block_class):
                if prompts is None:
                    raise ValueError("Prompts are required when use_prompts is True")
                x = block(x, prompts)
            else:
                x = block(x)

        return x


class DualPreNorm(nn.Module):
    """
    Dual pre-normalization layer for cross-attention mechanisms.
    
    Applies layer normalization to both input tensors before passing
    them to the underlying function.
    """
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class PriorAttentionBlock(nn.Module):
    """
    Prior attention block for knowledge transfer between features and priors.
    
    Implements bidirectional attention mechanism where priors are updated
    by aggregating from feature maps, and feature maps are enhanced by
    injecting knowledge from priors.
    """
    
    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head

        dim = feat_dim
        mlp_dim = dim * 4

        # Update priors by aggregating from the feature map
        self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

        # Update the feature map by injecting knowledge from the priors
        self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

    def forward(self, x1, x2):
        """
        Forward pass with bidirectional attention.
        
        Args:
            x1: Image feature map
            x2: Prior representations
            
        Returns:
            x1: Updated image features
            x2: Updated prior representations
        """
        # Update priors using image features
        x2 = self.prior_aggregate_block(x2, x1) + x2
        x2 = self.prior_ffn(x2) + x2

        # Update image features using priors
        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1

        return x1, x2


class PriorInitFusionLayer(nn.Module):
    """
    Prior initialization and fusion layer for multi-task and multi-modal learning.
    
    Initializes learnable task and modality priors and fuses them with
    image features through attention mechanisms.
    """
    
    def __init__(self, feat_dim, prior_dim, block_num=2, task_prior_num=42, modality_prior_num=2, l=10):
        super().__init__()
        
        # Randomly initialize the learnable priors
        self.task_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(task_prior_num+1, prior_dim))) # +1 for null token
        self.modality_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(modality_prior_num, l, prior_dim)))

        # Stack of attention layers for feature-prior interaction
        self.attn_layers = nn.ModuleList([])
        for i in range(block_num):
            self.attn_layers.append(PriorAttentionBlock(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0))

    def forward(self, x, tgt_idx, mod_idx):
        """
        Forward pass with prior selection and fusion.
        
        Args:
            x: Image feature map
            tgt_idx: Target task indices for each sample
            mod_idx: Modality indices for each sample
            
        Returns:
            x: Enhanced image features after prior fusion
            priors: Updated prior representations
        """
        B, C, D, H, W = x.shape
        
        # Select task and modality priors for each sample
        task_prior_list = []
        modality_prior_list = []
        for i in range(B):
            idxs = tgt_idx[i]
            task_prior_list.append(self.task_prior[idxs, :])
            modality_prior_list.append(self.modality_prior[mod_idx[i], :, :])

        task_priors = torch.stack(task_prior_list)
        modality_priors = torch.stack(modality_prior_list)
        modality_priors = modality_priors.squeeze(1)

        # Concatenate task and modality priors
        priors = torch.cat([task_priors, modality_priors], dim=1)
        
        # Reshape image features for attention computation
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = x.permute(0, 2, 1).contiguous()
        
        # Apply attention layers for feature-prior interaction
        for layer in self.attn_layers:
            x, priors = layer(x, priors)
        
        # Reshape back to original spatial dimensions
        x = x.permute(0, 2, 1)
        x = x.view(b, c, d, h, w).contiguous()

        return x, priors
