import torch
import pandas as pd
import sys
from pathlib import Path
import os
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess_pytorch import create_dataloader

def test_dataloader_debug():
    
    """Test that dataloader can load and return batches correctly"""
    
    print("Testing dataloader...")
    
    if not os.path.exists("data/processed/genbuster_cached") or \
       not os.path.exists("data/splits/train_metadata.csv"):
        pytest.skip("Skipping dataset-dependent test in CI")

    loader = create_dataloader(
        metadata_csv="data/splits/train_metadata.csv",      
        batch_size=4,                                        
        num_workers=0,                                       
        clip_mode=True,                                      
        shuffle=False,                                       
        cache_root="data/processed/genbuster_cached"         
    )
    
    print(f" Dataloader created")
    print(f"   Dataset size: {len(loader.dataset)}")
    
    # Get one batch
    frames, labels = next(iter(loader))
    
    print(f"\n Loaded one batch:")
    print(f"   Frames shape: {frames.shape}")    # [4, 3, 8, 224, 224]
    print(f"   Labels shape: {labels.shape}")    # [4]
    print(f"   Labels: {labels.tolist()}")
    
    # Assertions
    assert frames.ndim == 5, f"Expected 5D tensor, got {frames.ndim}D"
    assert frames.shape == torch.Size([4, 3, 8, 224, 224]), f"Unexpected shape: {frames.shape}"
    assert labels.ndim == 1, f"Expected 1D labels, got {labels.ndim}D"
    
    print(f"\n [PASS] All checks passed!")
    print(f"   Format: [Batch={frames.shape[0]}, Channels={frames.shape[1]}, Time={frames.shape[2]}, Height={frames.shape[3]}, Width={frames.shape[4]}]")

if __name__ == "__main__":
    test_dataloader_debug()