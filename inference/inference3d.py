"""
3D Inference functions - exact copy from HERMES for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference


def create_predictor(net, modality, dataset_name):
    """Creates a predictor function for sliding window inference that includes embedding and target indices"""
    def _predictor(input_tensor):
        # Get batch size of current window patches
        window_batch_size = input_tensor.shape[0]
        
        # Replicate embeddings and target indices for each patch in the window
        dataset_name_repeated = [dataset_name] * window_batch_size
        modality_repeated = modality.repeat(window_batch_size, 1)
        # Forward pass with replicated embeddings
        pred = net(input_tensor, modality_repeated, dataset_name_repeated, padd_classes=False)
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]
        return F.sigmoid(pred)
    return _predictor


def inference_whole_image(net, img, modality, dataset_name, args=None):
    '''
    img: torch tensor, B, C, D, H, W
    modality: modality embedding
    dataset_name: dataset name
    return: prob (after sigmoid), B, classes, D, H, W
    '''
    net.eval()
    with torch.no_grad():
        predictor = create_predictor(net, modality, dataset_name)
        # Use sliding window inference with full image as window size
        _, _, D, H, W = img.shape
        roi_size = (D, H, W)
        pred = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=predictor,
            overlap=0.0  # No overlap needed for whole image
        )
    return pred


def inference_sliding_window(net, img, modality, dataset_name, args):
    '''
    img: torch tensor, B, C, D, H, W
    modality: modality embedding  
    dataset_name: dataset name
    return: prob (after sigmoid), B, classes, D, H, W
    
    Uses MONAI's sliding window inference with configurable overlap
    '''
    net.eval()
    with torch.no_grad():
        predictor = create_predictor(net, modality, dataset_name)
        roi_size = args.window_size if hasattr(args, 'window_size') else (128, 128, 128)  # Default window size if not specified
        pred = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=1, #args.batch_size if hasattr(args, 'batch_size') else 4,  # Reduced for memory efficiency
            predictor=predictor,
            overlap=0.5,  # 50% overlap for better blending
            mode="gaussian",  # Use gaussian importance weighting
        )
    return pred


def split_idx(half_win, size, i): 
    ''' 
    half_win: The size of half window
    size: img size along one axis
    i: the patch index
    '''

    start_idx = half_win * i 
    end_idx = start_idx + half_win*2

    if end_idx > size:
        start_idx = size - half_win*2
        end_idx = size

    return start_idx, end_idx
