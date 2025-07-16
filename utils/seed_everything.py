from __future__ import division
import numpy as np
import torch
import os 
import random 

def set_seed(seed=42):
    """
    Set seed for reproducibility across random, numpy, torch (CPU & CUDA)
    """
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch (CPU)
    torch.cuda.manual_seed(seed)              # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)          # All GPUs (if multi-GPU)
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disable performance auto-tuner
    os.environ['PYTHONHASHSEED'] = str(seed)  # Hash seed for built-ins

# set_seed(42)  # Set your desired seed here