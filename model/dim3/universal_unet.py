"""
Universal Medical Image Segmentation UNet Architecture

This module implements a universal UNet architecture for multi-modal medical image segmentation.
The model supports different imaging modalities (CT, MRI, PET) and uses text-guided feature
modulation for universal segmentation across various anatomical structures.

Key Features:
- Multi-modal input processing (CT, MRI, PET)
- Text-guided feature modulation using ClipMD
- Dynamic convolution heads for class-specific segmentation
- FiLM (Feature-wise Linear Modulation) support
- LoRA fine-tuning for text encoder efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .universal_unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm, get_act
from utils import is_master  # Import at module level
import logging
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
import copy
import json
import random
import os

# Dataset-specific modality prompts for text guidance
DATASET_MODALITY_PROMPTS = {
    "bcv": "A computed tomography scan of the abdomen",
    "lits": "A computed tomography scan of the abdomen", 
    "kits": "A computed tomography scan of the abdomen",
    "amos_ct": "A computed tomography scan of the abdomen",
    "structseg_oar": "A computed tomography scan of the thorax",
    "amos_mr": "A Magnetic Resonance Imaging scan of the abdomen",
    "chaos&&t1_in": "A T1-weighted in-phase Magnetic Resonance Imaging scan of the abdomen",
    "chaos&&t1_out": "A T1-weighted out-of-phase Magnetic Resonance Imaging scan of the abdomen",
    "chaos&&t2": "A T2-weighted Magnetic Resonance Imaging scan of the abdomen", 
    "mnm": "A cine Magnetic Resonance Imaging scan of the cardiac region",
    "brain_structure": "A T1-weighted Magnetic Resonance Imaging scan of the brain",
    "autopet": "A positron emission tomography scan of the whole body",
    "hecktor_ct": "A computed tomography scan of the head and neck",
    "hecktor_pet": "A positron emission tomography scan of the head and neck",
}

# Dataset-specific class prompts for anatomical structures
DATASET_CLASS_PROMPTS = {
    "bcv": ['A computed tomography scan of a spleen', 'A computed tomography scan of a rightkidney', 'A computed tomography scan of a leftkidney', 'A computed tomography scan of a gallbladder', 'A computed tomography scan of a esophagus', 'A computed tomography scan of a liver', 'A computed tomography scan of a stomach', 'A computed tomography scan of a aorta', 'A computed tomography scan of a postcava', 'A computed tomography scan of a portal and splenic vein', 'A computed tomography scan of a pancreas', 'A computed tomography scan of a rightadrenal gland', 'A computed tomography scan of a leftadrenal gland'],
    "lits": ['A computed tomography scan of a liver', 'A computed tomography scan of a livertumor'],
    "kits": ['A computed tomography scan of right and left kidneys', 'A computed tomography scan of a kidneytumor'],
    "amos_ct": ['A computed tomography scan of a spleen', 'A computed tomography scan of a rightkidney', 'A computed tomography scan of a leftkidney', 'A computed tomography scan of a gallbladder', 'A computed tomography scan of a esophagus', 'A computed tomography scan of a liver', 'A computed tomography scan of a stomach', 'A computed tomography scan of a aorta', 'A computed tomography scan of a postcava', 'A computed tomography scan of a pancreas', 'A computed tomography scan of a rightadrenal gland', 'A computed tomography scan of a leftadrenal gland', 'A computed tomography scan of a duodenum', 'A computed tomography scan of a bladder', 'A computed tomography scan of a prostate or uterus'],
    "structseg_oar": ['A computed tomography scan of a rightlung', 'A computed tomography scan of a leftlung', 'A computed tomography scan of a heart', 'A computed tomography scan of a esophagus', 'A computed tomography scan of a trachea', 'A computed tomography scan of a spinalcord'],
    "structseg_head_oar": ['A computed tomography scan of a lefteye', 'A computed tomography scan of a righteye', 'A computed tomography scan of a leftlens', 'A computed tomography scan of a rightlens', 'A computed tomography scan of a leftoptical nerve', 'A computed tomography scan of a rightopticalnerve', 'A computed tomography scan of a opticalchiasma', 'A computed tomography scan of a pituitary', 'A computed tomography scan of a brainstem', 'A computed tomography scan of a lefttemporallobes', 'A computed tomography scan of a righttemporallobes', 'A computed tomography scan of a spinalcord', 'A computed tomography scan of a leftparotidgland', 'A computed tomography scan of a righparotidgland', 'A computed tomography scan of a leftinnerear', 'A computed tomography scan of a rightinnerear', 'A computed tomography scan of a leftmiddleear', 'A computed tomography scan of a rightmiddleear', 'A computed tomography scan of a lefttemporomandibular joint', 'A computed tomography scan of a righttemporomandibular joint', 'A computed tomography scan of a leftmandible', 'A computed tomography scan of a rightmandible'],
    "amos_mr": ['A magnetic resonance imaging scan of a spleen', 'A magnetic resonance imaging scan of a rightkidney', 'A magnetic resonance imaging scan of a leftkidney', 'A magnetic resonance imaging scan of a gallbladder', 'A magnetic resonance imaging scan of a esophagus', 'A magnetic resonance imaging scan of a liver', 'A magnetic resonance imaging scan of a stomach', 'A magnetic resonance imaging scan of a aorta', 'A magnetic resonance imaging scan of a postcava', 'A magnetic resonance imaging scan of a pancreas', 'A magnetic resonance imaging scan of a rightadrenal gland', 'A magnetic resonance imaging scan of a leftadrenal gland', 'A magnetic resonance imaging scan of a duodenum'],
    "chaos&&t1_in": ['A T1-weighted in-phase magnetic resonance imaging scan of a liver', 'A T1-weighted in-phase magnetic resonance imaging scan of a rightkidney', 'A T1-weighted in-phase magnetic resonance imaging scan of a leftkidney', 'A T1-weighted in-phase magnetic resonance imaging scan of a spleen'],
    "chaos&&t1_out": ['A T1-weighted out-phase magnetic resonance imaging scan of a liver', 'A T1-weighted out-phase magnetic resonance imaging scan of a rightkidney', 'A T1-weighted out-phase magnetic resonance imaging scan of a leftkidney', 'A T1-weighted out-phase magnetic resonance imaging scan of a spleen'],
    "chaos&&t2": ['A T2-weighted magnetic resonance imaging scan of a liver', 'A T2-weighted magnetic resonance imaging scan of a rightkidney', 'A T2-weighted magnetic resonance imaging scan of a leftkidney', 'A T2-weighted magnetic resonance imaging scan of a spleen'],
    "mnm": ['A magnetic resonance imaging scan of a leftventricle', 'A magnetic resonance imaging scan of a myocardium', 'A magnetic resonance imaging scan of a rightventricle'],
    "brain_structure": ['A magnetic resonance imaging scan of a graymatter', 'A magnetic resonance imaging scan of a whitematter', 'A magnetic resonance imaging scan of a csf'],
    "autopet": ['A positron emission tomography scan of a tumor'],
    "hecktor_ct": ["A computed tomography scan of a primary tumor","A computed tomography scan of a lymph nodes"],
    "hecktor_pet": ["A positron emission tomography scan of a primary tumor","A positron emission tomography scan of a lymph nodes"],
}

# Load augmented class prompts from JSON file
try:
    with open('dict_class_prompts_with_augmentations.json') as f:
        DICT_CLASS_PROMPTS_WITH_AUGMENTATIONS = json.load(f)
except FileNotFoundError:
    # Fallback to basic prompts if augmentation file not found
    DICT_CLASS_PROMPTS_WITH_AUGMENTATIONS = {k: [[prompt] for prompt in v] for k, v in DATASET_CLASS_PROMPTS.items()}


class Universal_UNet(nn.Module):
    """
    Universal UNet for multi-modal medical image segmentation.
    
    This architecture supports multiple imaging modalities (CT, MRI, PET) and uses
    text-guided feature modulation for universal segmentation across various datasets
    and anatomical structures.
    """
    
    def __init__(self, 
        in_ch, 
        base_ch, 
        scale=[2,2,2,2], 
        kernel_size=[3,3,3,3], 
        block='BasicBlock', 
        num_block=[2,2,2,2],
        pool=True, 
        norm='in',
        act='relu',
        num_prompts=10,
        prompt_dim=512,
        use_film=False,
        rank=0,
        clipmd_model_name="Idan0405/ClipMD",
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        fusion=False,
        ):
        super().__init__()
        """
        Initialize Universal UNet architecture.
        
        Args:
            in_ch: Number of input channels
            base_ch: Number of channels in the entry level
            scale: Downsample scale for each level
            kernel_size: 3D kernel size for each level
            block: Type of convolutional block ('BasicBlock', 'ConvNormAct')
            num_block: Number of blocks in each stage
            pool: Use maxpool or strided conv for downsampling
            norm: Normalization layer type ('bn', 'in', 'ln')
            act: Activation function type ('relu', 'gelu', 'elu')
            num_prompts: Number of text prompts
            prompt_dim: Dimension of text prompt embeddings
            use_film: Whether to use FiLM instead of dynamic convolution
            rank: Process rank for distributed training
            clipmd_model_name: Name of the ClipMD model for text encoding
            lora_r: Rank of LoRA layers for efficient fine-tuning
            lora_alpha: Alpha scaling for LoRA
            lora_dropout: Dropout probability for LoRA
            fusion: Whether to use fusion for different modalities
        """
        if rank == 0:
            logging.info(f"Initializing Universal_UNet with {'FiLM' if use_film else 'Dynamic Convolution'} modulation")
            
        self.base_ch = base_ch
        block = get_block(block)
        norm = get_norm(norm)
        act = get_act(act)

        self.max_n_classes = max([len(val) for val in DATASET_CLASS_PROMPTS.values()])
        
        # Initialize ClipMD text encoder and processor
        clipmd_model = AutoModel.from_pretrained(clipmd_model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(clipmd_model_name)
        
        # Extract only the text encoder part
        self.text_encoder = clipmd_model.text_model
        
        # Configure LoRA for efficient fine-tuning of text encoder
        lora_config = LoraConfig(
            r=32,  # Increased rank for more capacity
            lora_alpha=128,  # Increased alpha scaling (typically 4x the rank)
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.2,
            bias="lora_only",
            task_type="FEATURE_EXTRACTION",
            modules_to_save=["layer_norm", "encoder_layer_norm"],
        )
        
        # Apply LoRA to text encoder
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        
        # Freeze base model weights except LoRA parameters
        for param in self.text_encoder.base_model.parameters():
            param.requires_grad = False
        
        del clipmd_model  # Free up memory

        self.fusion = fusion
        
        # Modality-specific input convolution layers
        self.inc_ct = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.inc_pet = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.inc_mr = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm, act=act)

        # Encoder blocks with text-guided modulation
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block[0], block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block[1], block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block[2], block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block[3], block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)

        # Decoder blocks with text-guided modulation
        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block[2], block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block[1], block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block[0], block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm, act=act, use_prompts=True, use_film=use_film, prompt_dim=prompt_dim)
        self.up4 = up_block(2*base_ch, base_ch, num_block=2, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm, act=act, use_prompts=False, use_film=False, prompt_dim=prompt_dim)

        # Pre-classification feature reduction
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.ELU(),
            nn.Conv3d(base_ch, 8, kernel_size=1)
        )

        # Dynamic convolution parameters for class-specific heads
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)   # First conv layer
        weight_nums.append(8*8)   # Second conv layer
        weight_nums.append(8*1)   # Final conv layer
        bias_nums.append(8)       # First bias
        bias_nums.append(8)       # Second bias
        bias_nums.append(1)       # Final bias
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        # Controller for dynamic parameter generation
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), 
                                    kernel_size=1, stride=1, padding=0)
        
        # Text prompt processing components
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        # Global Average Pooling for encoder output
        self.gap = nn.Sequential(
                nn.GroupNorm(16, 10*base_ch),
                nn.ELU(),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(10*base_ch, 256, kernel_size=1, stride=1, padding=0)
            )

        # Text-to-vision feature projection
        self.text_to_vision = nn.Linear(512, 256)

        # Organ embedding for text prompt processing
        self.register_buffer('organ_embedding', torch.randn(1, 512))
        self.class_num = 1

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        """
        Parse the dynamic parameters for the dynamic convolution layers.
        
        Args:
            params: Dynamic parameters tensor
            channels: Number of feature channels
            weight_nums: List of weight sizes for each layer
            bias_nums: List of bias sizes for each layer
            
        Returns:
            weight_splits: List of weight tensors for each layer
            bias_splits: List of bias tensors for each layer
        """
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # For intermediate layers
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # For the final layer
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        """
        Apply the dynamic convolution layers using the generated parameters.
        
        Args:
            features: Input features to be processed
            weights: List of weights for each dynamic conv layer
            biases: List of biases for each dynamic conv layer
            num_insts: Number of instances (classes) being processed
            
        Returns:
            x: Output features after applying dynamic convolutions
        """
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.elu(x)
        return x

    def encode_text(self, text_prompts):
        """
        Encode text prompts using ClipMD text encoder with LoRA fine-tuning.
        
        Args:
            text_prompts: List of text prompts to encode
            
        Returns:
            text_embeddings: Tensor of shape (batch_size, 512) containing text embeddings
        """
        # Process text using ClipMD processor
        processed = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        # Move to correct device
        device = torch.device("cuda")
        
        # Extract only the input_ids and attention_mask
        model_inputs = {
            'input_ids': processed['input_ids'].to(device),
            'attention_mask': processed['attention_mask'].to(device)
        }
        
        # Get text embeddings from text encoder with error handling
        try:
            outputs = self.text_encoder(**model_inputs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # Fallback: try with only input_ids if attention_mask causes issues
                try:
                    outputs = self.text_encoder(input_ids=model_inputs['input_ids'])
                except TypeError:
                    # Final fallback: use the base model directly
                    outputs = self.text_encoder.base_model(**model_inputs)
            else:
                raise e
        
        # Extract embeddings - handle different output formats
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            text_embeddings = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use mean pooling if pooler_output is not available
            text_embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            # Direct tensor output
            text_embeddings = outputs
        
        return text_embeddings

    def forward(self, x, modality, datasets_names, padd_classes=True):
        """
        Forward pass of the Universal UNet.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, D, H, W)
            modality: Modality indices for each sample in the batch
            datasets_names: List of dataset names for each sample
            padd_classes: Whether to pad class prompts to max number of classes
            
        Returns:
            logits: Output segmentation logits
        """
        batch_size = x.shape[0]
        device = torch.device("cuda")
        
        # Prepare text prompts for each sample
        text_prompts = []
        text_class_prompts = []
        for dataset_name in datasets_names:
            # Add modality-specific prompts
            text_prompts.append(copy.deepcopy(DATASET_MODALITY_PROMPTS[dataset_name]))
            
            # Sample class prompts with augmentation
            sample_class_prompt = [random.choice(list_prompts) for list_prompts in DICT_CLASS_PROMPTS_WITH_AUGMENTATIONS[dataset_name]]
            text_class_prompts.append(copy.deepcopy(sample_class_prompt))
            
        # Pad class prompts to maximum number of classes if requested
        if padd_classes == True:
            for i in range(len(text_class_prompts)):
                diff = self.max_n_classes + 1 - len(text_class_prompts[i])
                text_class_prompts[i].extend([""] * diff)
        
        # Process modality text prompts and move to device
        text_mod_embeddings = self.encode_text(text_prompts).to(device)
        
        # Process each sample through its corresponding input convolution based on modality
        x1_list = []
        for i in range(batch_size):
            sample = x[i:i+1]  # Keep batch dimension
            mod_idx = modality[i].item()  # Get modality index for this sample
            
            # Route through corresponding inconv layer based on modality index
            if mod_idx == 0:  # CT
                x1_list.append(self.inc_ct(sample))
            elif mod_idx == 1:  # MRI
                x1_list.append(self.inc_mr(sample))
            elif mod_idx == 2:  # PET
                x1_list.append(self.inc_pet(sample))
            else:
                raise ValueError(f"Unknown modality index: {mod_idx}")
        
        # Combine processed samples back into a batch
        x1 = torch.cat(x1_list, dim=0)
        
        # Encoder path with text-guided modulation
        x2 = self.down1(x1, text_mod_embeddings)
        x3 = self.down2(x2, text_mod_embeddings)
        x4 = self.down3(x3, text_mod_embeddings)
        x5 = self.down4(x4, text_mod_embeddings)
        
        # Decoder path with text-guided modulation
        out = self.up1(x5, x4, text_mod_embeddings)
        out = self.up2(out, x3, text_mod_embeddings)
        out = self.up3(out, x2, text_mod_embeddings)
        out = self.up4(out, x1, text_mod_embeddings)
        
        # Global feature extraction from the deepest encoder feature
        x_feat = self.gap(x5)  # shape: (batch_size, 256, 1, 1, 1)
        
        # Process each sample with its specific class prompts
        logits_array = []
        for i in range(batch_size):
            # Process class prompts for this sample and move to device
            class_embeddings = self.encode_text(text_class_prompts[i]).to(device)
            
            # Process text prompt using text_to_vision projection
            task_encoding = F.elu(self.text_to_vision(class_embeddings))  # shape: (num_classes, 256)
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # shape: (num_classes, 256, 1, 1, 1)

            # Concatenate global features with text encoding
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(class_embeddings.shape[0], 1, 1, 1, 1), task_encoding], dim=1)  # shape: (num_classes, 512, 1, 1, 1)
            
            # Generate dynamic convolution parameters
            params = self.controller(x_cond)  # shape: (num_classes, total_params, 1, 1, 1)
            params = params.squeeze(-1).squeeze(-1).squeeze(-1)  # shape: (num_classes, total_params)
            
            # Prepare head inputs from decoder output
            head_inputs = self.precls_conv(out[i].unsqueeze(0))  # shape: (1, 8, D, H, W)
            head_inputs = head_inputs.repeat(class_embeddings.shape[0], 1, 1, 1, 1)  # shape: (num_classes, 8, D, H, W)
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W)
            
            # Parse dynamic parameters into weights and biases
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)
            
            # Apply dynamic convolution layers
            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))
        
        logits = torch.cat(logits_array, dim=0)

        return logits
