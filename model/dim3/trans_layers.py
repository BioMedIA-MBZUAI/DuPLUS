"""
Transformer Layers for Universal Medical Image Segmentation

This module contains transformer-based components including:
- Multi-head self-attention and cross-attention mechanisms
- Feed-forward networks (MLPs)
- Layer normalization variants
- Complete transformer blocks

These components are used for attention-based feature processing
and cross-modal feature fusion in the universal segmentation model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = [
    'Mlp',
    'Attention',
    'CrossAttention',
    'TransformerBlock',
    'LayerNorm',
    'PreNorm',
]


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network).
    
    Standard MLP with configurable hidden dimension, activation,
    and dropout for transformer blocks and other components.
    """
    
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper for transformer components.
    
    Applies layer normalization before the wrapped function,
    which is the standard approach in modern transformer architectures.
    """
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Implements the standard transformer self-attention with
    learnable query, key, and value projections.
    """
    
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Linear projections for Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def b_l_hd__b_h_l_d(self, x, heads):
        """Reshape from [B, L, H*D] to [B, H, L, D]"""
        b, l, n = x.shape
        h = heads
        d = int(n / h)
        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def b_h_l_d__b_l_hd(self, x):
        """Reshape from [B, H, L, D] to [B, L, H*D]"""
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x

    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor [B, L, C] where B=batch, L=sequence length, C=channels
            
        Returns:
            Output tensor after self-attention [B, L, C]
        """
        # Generate Q, K, V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])
        
        # Compute attention scores
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = self.b_h_l_d__b_l_hd(attned)
        
        # Final projection
        attned = self.to_out(attned)
        return attned


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.
    
    Implements cross-attention where queries come from one input
    and keys/values come from another, enabling feature fusion
    between different modalities or representations.
    """
    
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Separate projections for queries and key-values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def b_l_hd__b_h_l_d(self, x, heads):
        """Reshape from [B, L, H*D] to [B, H, L, D]"""
        b, l, n = x.shape
        h = heads
        d = int(n / h)
        x = x.view(b, l, h, d)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def b_h_l_d__b_l_hd(self, x):
        """Reshape from [B, H, L, D] to [B, L, H*D]"""
        b, h, l, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, l, -1).contiguous()
        return x

    def forward(self, x1, x2):
        """
        Forward pass of cross-attention.
        
        Args:
            x1: Query input tensor [B, L1, C]
            x2: Key-Value input tensor [B, L2, C]
            
        Returns:
            Output tensor [B, L1, C] - aggregates information from x2 to x1
        """
        # Generate Q from x1, K and V from x2
        q = self.to_q(x1)
        k, v = self.to_kv(x2).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q, k, v = map(lambda t: self.b_l_hd__b_h_l_d(t, self.heads), [q, k, v])

        # Compute cross-attention scores
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = self.b_h_l_d__b_l_hd(attned)

        # Final projection
        attned = self.to_out(attned)
        return attned


class TransformerBlock(nn.Module):
    """
    Complete transformer block with self-attention and feed-forward layers.
    
    Implements the standard transformer architecture with residual connections
    and pre-normalization for stable training.
    """
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.layers = nn.ModuleList([])

        # Stack multiple transformer layers
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
            ]))
            
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, L, C]
            
        Returns:
            Output tensor [B, L, C] after self-attention and feed-forward processing
        """
        for attn, ffn in self.layers:
            x = attn(x) + x  # Self-attention with residual connection
            x = ffn(x) + x   # Feed-forward with residual connection
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm supporting both channels_first and channels_last data formats.
    
    This is particularly useful for computer vision applications where
    feature maps can be in different formats (e.g., [B, C, H, W] vs [B, H, W, C]).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("LayerNorm data_format must be 'channels_last' or 'channels_first'")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        """
        Forward pass with format-aware normalization.
        
        Args:
            x: Input tensor in either channels_first or channels_last format
            
        Returns:
            Normalized tensor in the same format as input
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            # Manual layer norm for channels_first format (e.g., for 3D medical images)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Broadcast parameters for 3D case [B, C, D, H, W]
            x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
            return x
