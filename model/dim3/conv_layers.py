"""
Convolutional Layers for Universal Medical Image Segmentation

This module contains various convolutional building blocks including:
- Basic convolutional blocks with normalization and activation
- ResNet-style BasicBlock and Bottleneck blocks
- Feature modulation layers (FiLM and Dynamic Convolution)
- Efficient convolution variants (MobileNet-style blocks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .trans_layers import LayerNorm


__all__ = [
    'ConvNormAct',
    'BasicBlock', 
    'Bottleneck',
    'SingleConv',
    'DepthwiseSeparableConv',
    'FiLMLayer',
    'DynamicConv3D',
]


class ConvNormAct(nn.Module):
    """
    Basic building block grouping convolution, normalization and activation.
    
    This is the fundamental component used throughout the network architecture
    providing consistent conv-norm-act or preact-conv patterns.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.ELU, True, False]

        self.conv = nn.Conv3d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch, eps=1e-4) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch, eps=1e-4) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x): 
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))
        return out 


class DynamicConv3D(nn.Module):
    """
    Dynamic 3D convolution layer with text-guided parameter generation.
    
    Generates convolution weights and biases dynamically based on text prompts,
    allowing for adaptive feature processing conditioned on textual descriptions.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, prompt_dim):
        super(DynamicConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        # Project text prompts to convolution parameters
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels+1),           
        )

    def forward(self, x, prompts):
        """
        Apply dynamic convolution with text-guided parameters.
        
        Args:
            x: Input feature tensor [B, C, D, H, W]
            prompts: Text prompt embeddings [B, prompt_dim]
            
        Returns:
            Output tensor after dynamic convolution
        """
        batch_size, _, D, H, W = x.size()
        prompts = self.prompt_proj(prompts)

        out = []
        for i in range(batch_size):
            # Generate weights and biases from prompts
            weights = prompts[i][:-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            biases = prompts[i][-1].unsqueeze(0)
            
            # Apply dynamic convolution
            res = F.sigmoid(F.conv3d(x[i].view(1, -1, D, H, W), weights, bias=biases))
            out.append(res)

        return torch.cat(out, dim=0)
    

class SingleConv(nn.Module):
    """
    Single convolution layer with normalization and activation.
    
    Simplified wrapper around ConvNormAct for single convolution operations.
    """
    
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.ELU, True, False]

        pad_size = [i//2 for i in kernel_size]
        self.conv = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        return self.conv(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation to feature maps based on text prompts,
    providing an alternative to dynamic convolution for text-guided modulation.
    """
    
    def __init__(self, num_channels, prompt_dim, act=nn.ReLU):
        super().__init__()
        # Project prompts to generate gamma (scale) and beta (bias) parameters
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, 256),
            act(),
            nn.Linear(256, num_channels * 2)  # 2 for gamma and beta
        )
        
    def forward(self, x, prompts):
        """
        Apply FiLM modulation to input features.
        
        Args:
            x: Input feature tensor [B, C, D, H, W]
            prompts: Text prompt embeddings [B, prompt_dim]
            
        Returns:
            Modulated feature tensor
        """
        params = self.prompt_proj(prompts)  # [B, C*2]
        gamma, beta = torch.chunk(params, 2, dim=1)  # Each [B, C]      
        
        # Reshape for broadcasting over spatial dimensions
        gamma = gamma.view(-1, x.size(1), 1, 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1, 1)
        
        # Apply FiLM modulation: gamma * x + beta
        return gamma * x + beta


class BasicBlock(nn.Module):
    """
    ResNet-style BasicBlock with optional text-guided feature modulation.
    
    Implements residual connections with support for FiLM or dynamic convolution
    for text-guided feature adaptation.
    """
    
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, 
                 act=nn.ReLU, preact=True, use_dynamic_conv=False, use_film=False, prompt_dim=None):
        super().__init__()
        assert not (use_dynamic_conv and use_film), "Cannot use both dynamic convolution and FiLM simultaneously"
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.ELU, True, False]

        pad_size = [i//2 for i in kernel_size]

        # Main convolution path
        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, stride=1, padding=pad_size, norm=norm, act=act, preact=preact)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

        # Feature modulation components
        self.use_dynamic_conv = use_dynamic_conv
        self.use_film = use_film
        
        if use_dynamic_conv:
            self.dynamic_conv = DynamicConv3D(out_ch, out_ch, kernel_size=1, prompt_dim=prompt_dim)
        elif use_film:
            self.film = FiLMLayer(out_ch, prompt_dim, act=act)
            
        # Feature capture for analysis/debugging
        self.captured_features = {}
        self.capture_features = False

    def forward(self, x, prompts=None):
        """
        Forward pass with optional text-guided modulation.
        
        Args:
            x: Input feature tensor
            prompts: Text prompt embeddings (required if using modulation)
            
        Returns:
            Output feature tensor
        """
        residual = x

        # Main convolution path
        out = self.conv1(x)
        out = self.conv2(out)

        # Residual connection
        out = out + self.shortcut(residual)

        # Capture features before modulation
        if self.capture_features:
            self.captured_features['before_modulation'] = out.detach().cpu().clone()

        # Apply text-guided modulation
        if self.use_dynamic_conv and prompts is not None:
            dyn_out = self.dynamic_conv(out, prompts)
            out = out * dyn_out  # Element-wise modulation
        elif self.use_film and prompts is not None:
            out = self.film(out, prompts)  # FiLM modulation
            
            # Capture features after modulation
            if self.capture_features:
                self.captured_features['after_modulation'] = out.detach().cpu().clone()
                
        elif (self.use_film or self.use_dynamic_conv) and prompts is None:
            raise ValueError("Prompts are required when using text-guided modulation")
        
        return out
        
    def enable_feature_capture(self):
        """Enable feature capture for analysis"""
        self.capture_features = True
        self.captured_features = {}
        
    def disable_feature_capture(self):
        """Disable feature capture"""
        self.capture_features = False
        self.captured_features = {}
        
    def get_captured_features(self):
        """Get captured features for analysis"""
        return self.captured_features


class Bottleneck(nn.Module):
    """
    ResNet-style Bottleneck block for more efficient computation.
    
    Uses 1x1 -> 3x3 -> 1x1 convolution pattern to reduce parameters
    while maintaining representational capacity.
    """
    
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, groups=1, dilation=1, 
                 norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.ELU, True, False]

        pad_size = [i//2 for i in kernel_size]
        
        self.expansion = 2
        
        # Bottleneck architecture: 1x1 -> 3x3 -> 1x1
        self.conv1 = ConvNormAct(in_ch, out_ch//self.expansion, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//self.expansion, out_ch//self.expansion, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)
        self.conv3 = ConvNormAct(out_ch//self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)
        return out


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient computation.
    
    Separates spatial and channel-wise convolutions to reduce
    computational complexity while maintaining expressiveness.
    """
    
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, list):
            padding = [i//2 for i in kernel_size]
        else:
            padding = kernel_size // 2

        # Depthwise convolution (spatial processing per channel)
        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,  # Key: groups=in_ch for depthwise
            bias=bias
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Applies adaptive channel-wise weighting based on global
    feature statistics to improve representational quality.
    """
    
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
                        nn.Conv3d(in_ch, in_ch//ratio, kernel_size=1),
                        act(),
                        nn.Conv3d(in_ch//ratio, in_ch, kernel_size=1),
                        nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)
        return x * out


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) regularization.
    
    Randomly drops residual connections during training
    to improve generalization and reduce overfitting.
    """
    
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask
        return x
