import os
import sys
import random
import numpy as np
import torch
import logging
import yaml

def get_random_seed(config_path="config.yaml"):
    """Return reproducible random seed from config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        seed = config.get("seed", 42)
        return seed
    except FileNotFoundError:
        logging.warning("config.yaml not found. Using default seed: 42")
        return 42
    

def setup_environment(config_path="config.yaml"):
    """Setup environment for reproducible PyTorch training."""
    
    # CRITICAL: Add DLL paths FIRST before any other imports
    _setup_dll_paths()
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    mixed_precision = config.get("mixed_precision", True)
    log_level = config.get("log_level", "INFO")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Mixed precision for PyTorch
    if mixed_precision:
        # Enable TF32 on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU detected: {gpu_name}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.warning("No GPU detected, running on CPU")
    
    info_msg = f"Environment set. Seed={seed}, Mixed precision={mixed_precision}"
    logging.info(info_msg)
    
    return config


def _setup_dll_paths():
    """Setup DLL search paths for PyTorch (Windows only)"""
    if sys.platform != 'win32':
        return
    
    # MKL DLLs location
    library_bin = os.path.join(sys.prefix, 'Library', 'bin')
    
    # PyTorch lib location
    torch_lib = os.path.join(sys.prefix, 'lib', 'site-packages', 'torch', 'lib')
    
    # CUDA location
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    
    paths_to_add = []
    
    for path in [library_bin, torch_lib, cuda_bin]:
        if os.path.exists(path):
            paths_to_add.append(path)
    
    # Update PATH
    if paths_to_add:
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + current_path
    
    # For Python 3.8+
    if sys.version_info >= (3, 8) and hasattr(os, 'add_dll_directory'):
        for path in paths_to_add:
            try:
                os.add_dll_directory(path)
            except (FileNotFoundError, OSError):
                pass