# src/utils/cache_manager.py

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
    Manages .pt tensor caching with integrity verification and atomic operations.
    
    Features:
    - Atomic write operations (temp file + rename)
    - SHA256 checksum verification
    - Metadata tracking (shape, dtype, version)
    - Backward-compatible [T, H, W, C] format
    - Future-ready [C, T, H, W] format support
    """
    
    VERSION = "2.0.0"
    SUPPORTED_FORMATS = ["THWC", "CTHW"]  # Backward compatible / Optimized
    DEFAULT_FORMAT = "THWC"  # For backward compatibility
    
    def __init__(self, cache_root: str, format_type: str = "THWC"):
        """
        Initialize cache manager.
        
        Args:
            cache_root: Root directory for cached tensors
            format_type: "THWC" (backward compatible) or "CTHW" (optimized)
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        if format_type not in self.SUPPORTED_FORMATS:
            raise ValueError(f"format_type must be one of {self.SUPPORTED_FORMATS}")
        self.format_type = format_type
        
        logger.info(f"CacheManager initialized: {cache_root} (format: {format_type})")
    
    @staticmethod
    def compute_checksum(tensor: torch.Tensor) -> str:
        """
        Compute SHA256 checksum of tensor data.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Hex string of SHA256 hash
        """
        # Use tensor bytes for checksum
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    @staticmethod
    def compute_file_checksum(file_path: Path) -> str:
        """
        Compute SHA256 checksum of file contents.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex string of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def save(self, 
             tensor: torch.Tensor, 
             output_path: Path,
             metadata: Optional[Dict[str, Any]] = None,
             verify: bool = True) -> bool:
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            save_metadata = {
                'version': self.VERSION,
                'format': self.format_type,
                'shape': list(tensor.shape),
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
            
            # Verify write if requested
            if verify:
                file_checksum = self.compute_file_checksum(tmp_path)
                save_metadata['file_checksum'] = file_checksum
            
            # Atomic rename
            shutil.move(str(tmp_path), str(output_path))
            
            logger.debug(f"Saved: {output_path} (shape: {tensor.shape})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            # Cleanup temp file if exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            return False
    
    def load(self, 
             file_path: Path,
             verify: bool = True,
             device: str = 'cpu') -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Load data
            data = torch.load(file_path, map_location=device, weights_only=False)
            
            # Handle legacy format (raw tensor without metadata)
            if isinstance(data, torch.Tensor):
                logger.warning(f"Legacy format detected: {file_path}")
                return data, {'version': '1.0.0', 'format': 'unknown'}
            
            tensor = data['tensor']
            metadata = data.get('metadata', {})
            
            # Verify checksum if available and requested
            if verify and 'checksum' in metadata:
                computed = self.compute_checksum(tensor)
                stored = metadata['checksum']
                
                if computed != stored:
                    logger.error(f"Checksum mismatch: {file_path}")
                    logger.error(f"  Expected: {stored}")
                    logger.error(f"  Got: {computed}")
                    return None
            
            logger.debug(f"Loaded: {file_path} (shape: {tensor.shape})")
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def validate_cache_file(self, file_path: Path) -> Dict[str, Any]:
        
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
                result['error'] = 'Failed to load or verify'
                return result
            
            tensor, metadata = loaded
            
            # Check tensor properties
            if tensor.ndim != 4:
                result['error'] = f'Invalid dimensions: {tensor.ndim} (expected 4)'
                return result
            
            # Validate format
            expected_format = metadata.get('format', 'unknown')
            if expected_format not in self.SUPPORTED_FORMATS:
                result['error'] = f'Unknown format: {expected_format}'
                return result
            
            result['valid'] = True
            result['metadata'] = metadata
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        
        pt_files = list(self.cache_root.rglob('*.pt'))
        
        stats = {
            'total_files': len(pt_files),
            'total_size_gb': sum(f.stat().st_size for f in pt_files) / (1024**3),
            'format_counts': {},
            'corrupted': 0
        }
        
        for pt_file in pt_files:
            validation = self.validate_cache_file(pt_file)
            
            if validation['valid']:
                fmt = validation['metadata'].get('format', 'unknown')
                stats['format_counts'][fmt] = stats['format_counts'].get(fmt, 0) + 1
            else:
                stats['corrupted'] += 1
        
        return stats


class CacheConfig:
    """Configuration for caching operations."""
    
    def __init__(self,
                 source_root: str = "D:/GenBuster200k/processed/frames",
                 cache_root: str = "C:/Personal project/ai_video_detector/data/processed/cached",
                 format_type: str = "THWC",
                 num_frames: int = 16,
                 resolution: int = 224,
                 normalize: bool = True,
                 num_workers: int = 8,
                 batch_size: int = 100,
                 verify_integrity: bool = True):
        """
        Initialize cache configuration.
        
        Args:
            source_root: Root directory of extracted frames
            cache_root: Root directory for cached tensors
            format_type: "THWC" or "CTHW"
            num_frames: Expected frames per video
            resolution: Frame resolution (H=W)
            normalize: Whether to normalize to [0,1] float32
            num_workers: Parallel workers for caching
            batch_size: Checkpoint frequency
            verify_integrity: Enable checksum verification
        """
        self.source_root = Path(source_root)
        self.cache_root = Path(cache_root)
        self.format_type = format_type
        self.num_frames = num_frames
        self.resolution = resolution
        self.normalize = normalize
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verify_integrity = verify_integrity
        
        # Validate paths
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        
        self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dict."""
        return {
            'source_root': str(self.source_root),
            'cache_root': str(self.cache_root),
            'format_type': self.format_type,
            'num_frames': self.num_frames,
            'resolution': self.resolution,
            'normalize': self.normalize,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'verify_integrity': self.verify_integrity
        }
    
    def save(self, output_path: Path):
        """Save configuration to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: Path) -> 'CacheConfig':
        """Load configuration from JSON."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Convenience functions
def save_cached_tensor(tensor: torch.Tensor, 
                       output_path: Path,
                       format_type: str = "THWC",
                       verify: bool = True) -> bool:
    
    manager = CacheManager(output_path.parent, format_type=format_type)
    return manager.save(tensor, output_path, verify=verify)


def load_cached_tensor(file_path: Path,
                       verify: bool = True,
                       device: str = 'cpu') -> Optional[torch.Tensor]:
    
    manager = CacheManager(file_path.parent)
    result = manager.load(file_path, verify=verify, device=device)
    
    if result is None:
        return None
    
    tensor, _ = result
    return tensor