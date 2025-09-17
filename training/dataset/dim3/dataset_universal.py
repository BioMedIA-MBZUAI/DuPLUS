# UniversalDataset for 3D medical image segmentation
# Anonymized for WACV2026 submission

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from training import augmentation
from .dataset_config import train_test_split, dataset_lab_map, dataset_modality_map, dataset_sample_weight, dataset_aug_prob

# ...existing code...

class UniversalDataset(Dataset):
    # ...existing code...
