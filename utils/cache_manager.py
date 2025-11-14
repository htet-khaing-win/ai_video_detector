"""
CTHW-Native Cache Management System for Video Tensors

Standard Format: [C, T, H, W] = [3, 16, 224, 224]
  - C: RGB channels (3)
  - T: Temporal frames (16)
  - H: Height (224)
  - W: Width (224)

This format is optimal for:
  - Direct GPU loading (no permutation needed)
  - PyTorch video models (X3D, SlowFast, I3D)
  - Memory layout aligned with NCHW convolutions

Version: 3.0.0 (CTHW Native)
"""

import torch
import hashlib
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages .pt tensor caching in CTHW format with integrity verification.
    
    All tensors MUST be in [C, T, H, W] format.
    """
    
    VERSION = "3.0.0"
    TENSOR_FORMAT = "CTHW"  # Standardized format
    EXPECTED_DIMS = 4  # [C, T, H, W]
    EXPECTED_CHANNELS = 3  # RGB
    
    def __init__(self, 
                 cache_root: str,
                 num_frames: int = 16,
                 resolution: int = 224):
        """
        Initialize CTHW-native cache manager.
        
        Args:
            cache_root: Root directory for cached tensors
            num_frames: Expected temporal frames (T dimension)
            resolution: Expected spatial resolution (H=W)
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.resolution = resolution
        
        self.expected_shape = (self.EXPECTED_CHANNELS, num_frames, resolution, resolution)
        
        logger.info(
            f"CacheManager initialized (CTHW format)\n"
            f"  Root: {cache_root}\n"
            f"  Expected shape: {self.expected_shape}"
        )
    
    @staticmethod
    def compute_checksum(tensor: torch.Tensor) -> str:
        """Compute SHA256 checksum of tensor data."""
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    def validate_tensor_format(self, tensor: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate tensor is in correct CTHW format.
        
        Args:
            tensor: Input tensor to validate
            
        Returns:
            (is_valid, error_message)
        """
        # Check dimensions
        if tensor.ndim != self.EXPECTED_DIMS:
            return False, f"Expected {self.EXPECTED_DIMS}D tensor, got {tensor.ndim}D"
        
        # Check shape matches [C, T, H, W]
        if tensor.shape != self.expected_shape:
            return False, (
                f"Shape mismatch: expected {self.expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        
        # Check channels
        if tensor.shape[0] != self.EXPECTED_CHANNELS:
            return False, f"Expected {self.EXPECTED_CHANNELS} channels, got {tensor.shape[0]}"
        
        # Check dtype (should be uint8 or float32)
        if tensor.dtype not in (torch.uint8, torch.float32):
            return False, f"Invalid dtype: {tensor.dtype} (expected uint8 or float32)"
        
        # Check value range
        if tensor.dtype == torch.uint8:
            if tensor.min() < 0 or tensor.max() > 255:
                return False, f"uint8 values out of range: [{tensor.min()}, {tensor.max()}]"
        elif tensor.dtype == torch.float32:
            if tensor.min() < 0 or tensor.max() > 1:
                return False, f"float32 values out of range: [{tensor.min():.3f}, {tensor.max():.3f}]"
        
        return True, ""
    
    def save(self, 
             tensor: torch.Tensor,
             output_path: Path,
             metadata: Optional[Dict[str, Any]] = None,
             verify: bool = True) -> bool:
        """
        Save tensor in CTHW format with validation and integrity checks.
        
        Args:
            tensor: Tensor in [C, T, H, W] format
            output_path: Destination file path
            metadata: Optional metadata dict
            verify: Whether to verify write integrity
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # CRITICAL: Validate tensor format
            is_valid, error_msg = self.validate_tensor_format(tensor)
            if not is_valid:
                logger.error(f"Tensor validation failed: {error_msg}")
                return False
            
            # Prepare metadata
            save_metadata = {
                'version': self.VERSION,
                'format': self.TENSOR_FORMAT,
                'shape': list(tensor.shape),
                'shape_names': ['C', 'T', 'H', 'W'],
                'dtype': str(tensor.dtype),
                'checksum': self.compute_checksum(tensor) if verify else None,
            }
            
            # Merge with user metadata
            if metadata:
                save_metadata.update(metadata)
            
            # Package data
            save_data = {
                'tensor': tensor,
                'metadata': save_metadata
            }
            
            # Atomic write: temp file + rename
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=output_path.parent,
                suffix='.tmp'
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                torch.save(save_data, tmp_file)
            
            # Atomic rename
            shutil.move(str(tmp_path), str(output_path))
            
            logger.debug(
                f"Saved: {output_path.name} "
                f"(shape: {tensor.shape}, dtype: {tensor.dtype})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            # Cleanup temp file if exists
            if 'tmp_path' in locals() and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass
            return False
    
    def load(self,
             file_path: Path,
             verify: bool = True,
             device: str = 'cpu') -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Load tensor from .pt file with format validation.
        
        Args:
            file_path: Path to .pt file
            verify: Whether to verify checksum
            device: Device to load tensor to
            
        Returns:
            (tensor, metadata) tuple, or None if load fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Load data
            data = torch.load(file_path, map_location=device, weights_only=False)
            
            # Handle legacy formats (auto-convert or reject)
            if isinstance(data, torch.Tensor):
                logger.error(
                    f"Legacy format detected (raw tensor): {file_path}\n"
                    f"  This file needs to be re-cached in CTHW format."
                )
                return None
            
            tensor = data['tensor']
            metadata = data.get('metadata', {})
            
            # CRITICAL: Validate format
            stored_format = metadata.get('format', 'unknown')
            if stored_format != self.TENSOR_FORMAT:
                logger.error(
                    f"Format mismatch in {file_path}:\n"
                    f"  Expected: {self.TENSOR_FORMAT}\n"
                    f"  Got: {stored_format}\n"
                    f"  This file needs to be re-cached."
                )
                return None
            
            # Validate tensor shape
            is_valid, error_msg = self.validate_tensor_format(tensor)
            if not is_valid:
                logger.error(f"Invalid cached tensor {file_path}: {error_msg}")
                return None
            
            # Verify checksum if available and requested
            if verify and 'checksum' in metadata:
                computed = self.compute_checksum(tensor)
                stored = metadata['checksum']
                
                if computed != stored:
                    logger.error(
                        f"Checksum mismatch: {file_path}\n"
                        f"  Expected: {stored}\n"
                        f"  Got: {computed}"
                    )
                    return None
            
            logger.debug(
                f"Loaded: {file_path.name} "
                f"(shape: {tensor.shape}, dtype: {tensor.dtype})"
            )
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def validate_cache_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a cache file and return diagnostic info.
        
        Args:
            file_path: Path to .pt file
            
        Returns:
            Dict with validation results
        """
        result = {
            'path': str(file_path),
            'exists': file_path.exists(),
            'valid': False,
            'error': None,
            'metadata': None
        }
        
        if not result['exists']:
            result['error'] = 'File does not exist'
            return result
        
        try:
            loaded = self.load(file_path, verify=True)
            
            if loaded is None:
                result['error'] = 'Failed to load or invalid format'
                return result
            
            tensor, metadata = loaded
            
            # All checks passed
            result['valid'] = True
            result['metadata'] = metadata
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        pt_files = list(self.cache_root.rglob('*.pt'))
        
        stats = {
            'total_files': len(pt_files),
            'total_size_gb': sum(f.stat().st_size for f in pt_files) / (1024**3),
            'valid': 0,
            'invalid': 0,
            'legacy_format': 0,
        }
        
        for pt_file in pt_files:
            validation = self.validate_cache_file(pt_file)
            
            if validation['valid']:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                if 'format mismatch' in validation.get('error', '').lower():
                    stats['legacy_format'] += 1
        
        stats['valid_percentage'] = (stats['valid'] / len(pt_files) * 100) if pt_files else 0
        
        return stats


class CacheConfig:
    """Configuration for CTHW-native caching operations."""
    
    def __init__(self,
                 source_root: str = "D:/GenBuster200k/processed/frames",
                 cache_root: str = "D:/GenBuster200k/processed/cached",
                 num_frames: int = 16,
                 resolution: int = 224,
                 normalize: bool = False,
                 num_workers: int = 4,
                 batch_size: int = 50,
                 verify_integrity: bool = True):
        """
        Initialize cache configuration for CTHW format.
        
        Args:
            source_root: Root directory of extracted frames
            cache_root: Root directory for cached tensors (NEW LOCATION)
            num_frames: Expected frames per video (T dimension)
            resolution: Frame resolution (H=W dimensions)
            normalize: Whether to normalize to [0,1] float32
            num_workers: Parallel workers for caching
            batch_size: Checkpoint frequency
            verify_integrity: Enable checksum verification
        """
        self.source_root = Path(source_root)
        self.cache_root = Path(cache_root)
        self.num_frames = num_frames
        self.resolution = resolution
        self.normalize = normalize
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verify_integrity = verify_integrity
        
        # Tensor format specification
        self.tensor_format = "CTHW"
        self.expected_shape = (3, num_frames, resolution, resolution)
        
        # Validate paths
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        
        self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dict."""
        return {
            'source_root': str(self.source_root),
            'cache_root': str(self.cache_root),
            'tensor_format': self.tensor_format,
            'expected_shape': self.expected_shape,
            'num_frames': self.num_frames,
            'resolution': self.resolution,
            'normalize': self.normalize,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'verify_integrity': self.verify_integrity,
            'version': '3.0.0'
        }
    
    def save(self, output_path: Path):
        """Save configuration to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: Path) -> 'CacheConfig':
        """Load configuration from JSON."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove non-constructor keys
        config_dict.pop('tensor_format', None)
        config_dict.pop('expected_shape', None)
        config_dict.pop('version', None)
        
        return cls(**config_dict)


# Convenience functions
def save_cached_tensor(tensor: torch.Tensor,
                       output_path: Path,
                       num_frames: int = 16,
                       resolution: int = 224,
                       verify: bool = True) -> bool:
    """
    Save a tensor in CTHW format.
    
    Args:
        tensor: Tensor in [C, T, H, W] format
        output_path: Destination path
        num_frames: Expected T dimension
        resolution: Expected H=W dimensions
        verify: Enable verification
        
    Returns:
        Success status
    """
    manager = CacheManager(
        output_path.parent,
        num_frames=num_frames,
        resolution=resolution
    )
    return manager.save(tensor, output_path, verify=verify)


def load_cached_tensor(file_path: Path,
                       num_frames: int = 16,
                       resolution: int = 224,
                       verify: bool = True,
                       device: str = 'cpu') -> Optional[torch.Tensor]:
    """
    Load a CTHW tensor.
    
    Args:
        file_path: Path to .pt file
        num_frames: Expected T dimension
        resolution: Expected H=W dimensions
        verify: Enable verification
        device: Target device
        
    Returns:
        Loaded tensor in [C, T, H, W] format or None
    """
    manager = CacheManager(
        file_path.parent,
        num_frames=num_frames,
        resolution=resolution
    )
    result = manager.load(file_path, verify=verify, device=device)
    
    if result is None:
        return None
    
    tensor, _ = result
    return tensor