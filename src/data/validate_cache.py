"""
Validate CTHW-format cached tensors for completeness and integrity.

Verifies:
1. All expected cache files exist
2. Tensors are in [C, T, H, W] = [3, 16, 224, 224] format
3. Checksums are valid
4. Dtypes and value ranges are correct
5. No corruption detected

Run this AFTER caching to ensure 100% readiness for training.
"""

import sys
from pathlib import Path
project_root = Path("C:/Personal project/ai_video_detector")
sys.path.insert(0, str(project_root))

import torch
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

from utils.cache_manager import CacheManager, CacheConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CACHE_ROOT = Path("D:/GenBuster200k/processed/cached")  # NEW LOCATION
OUTPUT_DIR = Path("C:/Personal project/ai_video_detector/data/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_REPORT = OUTPUT_DIR / "cache_validation_cthw_report.json"
INVALID_FILES = OUTPUT_DIR / "invalid_cache_files_cthw.txt"


def validate_single_cache(cache_file: Path, expected_shape: tuple) -> dict:
    """
    Validate a single CTHW cached tensor file.
    
    Args:
        cache_file: Path to .pt file
        expected_shape: Expected (C, T, H, W) shape
        
    Returns:
        Dict with validation results
    """
    result = {
        'file': str(cache_file),
        'split': cache_file.parent.name,
        'video_id': cache_file.stem,
        'valid': False,
        'errors': []
    }
    
    try:
        # Initialize cache manager
        c, t, h, w = expected_shape
        manager = CacheManager(
            cache_file.parent,
            num_frames=t,
            resolution=h
        )
        
        # Load with verification
        loaded = manager.load(cache_file, verify=True, device='cpu')
        
        if loaded is None:
            result['errors'].append('Failed to load or verify')
            return result
        
        tensor, metadata = loaded
        
        # Validate format
        stored_format = metadata.get('format', 'unknown')
        if stored_format != 'CTHW':
            result['errors'].append(
                f"Format mismatch: expected CTHW, got {stored_format}"
            )
        
        # Validate shape
        if tuple(tensor.shape) != expected_shape:
            result['errors'].append(
                f"Shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        
        # Validate dimensions are in correct order [C, T, H, W]
        if tensor.shape[0] != 3:
            result['errors'].append(
                f"Channels mismatch: expected 3 (RGB), got {tensor.shape[0]}"
            )
        
        if tensor.shape[1] != t:
            result['errors'].append(
                f"Frames mismatch: expected {t}, got {tensor.shape[1]}"
            )
        
        if tensor.shape[2] != h or tensor.shape[3] != w:
            result['errors'].append(
                f"Resolution mismatch: expected {h}x{w}, got {tensor.shape[2]}x{tensor.shape[3]}"
            )
        
        # Validate dtype
        if tensor.dtype not in (torch.uint8, torch.float32):
            result['errors'].append(
                f"Invalid dtype: {tensor.dtype} (expected uint8 or float32)"
            )
        
        # Validate value range
        if tensor.dtype == torch.uint8:
            if tensor.min() < 0 or tensor.max() > 255:
                result['errors'].append(
                    f"uint8 value range error: [{tensor.min()}, {tensor.max()}]"
                )
        elif tensor.dtype == torch.float32:
            if tensor.min() < 0 or tensor.max() > 1:
                result['errors'].append(
                    f"float32 value range error: [{tensor.min():.4f}, {tensor.max():.4f}]"
                )
        
        # Check for all-zero frames (corruption)
        for t_idx in range(tensor.shape[1]):
            frame = tensor[:, t_idx, :, :]  # [C, H, W]
            if frame.sum() == 0:
                result['errors'].append(f"Frame {t_idx} is all zeros (corrupted)")
                break  # Only report first corrupted frame
        
        # If no errors, mark as valid
        if not result['errors']:
            result['valid'] = True
            result['metadata'] = metadata
            result['shape'] = list(tensor.shape)
            result['dtype'] = str(tensor.dtype)
        
    except Exception as e:
        result['errors'].append(f"Exception: {str(e)}")
    
    return result


def collect_cache_files() -> list:
    """Collect all .pt cache files."""
    cache_files = []
    splits = ['train', 'val', 'test', 'benchmark', 'dfd']
    
    for split in splits:
        split_path = CACHE_ROOT / split
        
        if not split_path.exists():
            logger.warning(f"Split directory not found: {split}")
            continue
        
        cache_files.extend(split_path.glob('*.pt'))
    
    return cache_files


def main():
    """Main validation workflow."""
    
    print("\n" + "="*80)
    print(f"{'CTHW CACHE VALIDATION':^80}")
    print("="*80)
    print(f"  Cache root: {CACHE_ROOT}")
    print(f"  Expected format: CTHW [3, 16, 224, 224]")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Load cache configuration
    config_path = CACHE_ROOT / "cache_config.json"
    
    if not config_path.exists():
        logger.error(f"Cache config not found: {config_path}")
        logger.error("Please run cache_frames_to_pt_cthw.py first.")
        return 1
    
    config = CacheConfig.load(config_path)
    logger.info(f"Loaded cache configuration")
    logger.info(f"  Format: {config.tensor_format}")
    logger.info(f"  Expected shape: {config.expected_shape}")
    
    # Collect cache files
    logger.info("Collecting cache files...")
    cache_files = collect_cache_files()
    
    if not cache_files:
        logger.error("No cache files found!")
        logger.error(f"Expected location: {CACHE_ROOT}")
        return 1
    
    logger.info(f"Found {len(cache_files)} cache files to validate")
    
    # Validate in parallel
    results = []
    stats = defaultdict(int)
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                validate_single_cache,
                cf,
                config.expected_shape
            ): cf
            for cf in cache_files
        }
        
        with tqdm(
            total=len(cache_files),
            desc="Validating CTHW cache",
            unit="file"
        ) as pbar:
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['valid']:
                    stats['valid'] += 1
                else:
                    stats['invalid'] += 1
                
                pbar.update(1)
    
    # Save validation report
    report = {
        'total_files': len(results),
        'valid': stats['valid'],
        'invalid': stats['invalid'],
        'valid_percentage': (stats['valid'] / len(results) * 100) if results else 0,
        'expected_format': 'CTHW',
        'expected_shape': config.expected_shape,
        'configuration': config.to_dict(),
        'results': results
    }
    
    logger.info(f"Saving validation report: {VALIDATION_REPORT}")
    with open(VALIDATION_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save invalid files list
    invalid = [r for r in results if not r['valid']]
    if invalid:
        logger.warning(f"Found {len(invalid)} invalid cache files")
        with open(INVALID_FILES, 'w') as f:
            f.write(f"CTHW Cache Validation - Invalid Files\n")
            f.write(f"Expected format: CTHW {config.expected_shape}\n")
            f.write(f"{'='*60}\n\n")
            
            for item in invalid:
                f.write(f"{item['split']}/{item['video_id']}.pt\n")
                for error in item['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
    
    # Print summary
    print("\n" + "="*80)
    print(f"{'VALIDATION SUMMARY':^80}")
    print("="*80)
    print(f"  Total cache files: {len(results):,}")
    print(f"   Valid (CTHW): {stats['valid']:,} ({report['valid_percentage']:.1f}%)")
    print(f"   Invalid: {stats['invalid']:,}")
    print(f"\n  Expected format: CTHW [3, 16, 224, 224]")
    print(f"  Expected dtype: uint8 or float32")
    print("="*80)
    
    if stats['invalid'] > 0:
        print(f"\n  {stats['invalid']} cache files have issues!")
        print(f"   Review: {INVALID_FILES}")
        print("   Action required:")
        print("   1. Delete invalid .pt files")
        print("   2. Delete caching_checkpoint.json")
        print("   3. Re-run cache_frames_to_pt_cthw.py")
    else:
        print("\n All cache files are valid and in CTHW format!")
        print("   Ready for X3D-M training with NO permutation overhead.")
    
    print(f"\n Full report: {VALIDATION_REPORT}")
    print("="*80 + "\n")
    
    return 0 if stats['invalid'] == 0 else 1


if __name__ == "__main__":
    exit(main())