# validate_cache.py

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
CACHE_ROOT = Path("C:/Personal project/ai_video_detector/data/processed/cached")
OUTPUT_DIR = Path("C:/Personal project/ai_video_detector/data/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_REPORT = OUTPUT_DIR / "cache_validation_report.json"
INVALID_FILES = OUTPUT_DIR / "invalid_cache_files.txt"


def validate_single_cache(cache_file: Path, expected_config: dict) -> dict:
    
    result = {
        'file': str(cache_file),
        'split': cache_file.parent.name,
        'video_id': cache_file.stem,
        'valid': False,
        'errors': []
    }
    
    try:
        # Initialize cache manager
        manager = CacheManager(cache_file.parent)
        
        # Load with verification
        loaded = manager.load(cache_file, verify=True, device='cpu')
        
        if loaded is None:
            result['errors'].append('Failed to load or verify checksum')
            return result
        
        tensor, metadata = loaded
        
        # Validate shape
        expected_shape = [
            expected_config['num_frames'],
            expected_config['resolution'],
            expected_config['resolution'],
            3  # RGB channels
        ]
        
        if list(tensor.shape) != expected_shape:
            result['errors'].append(
                f"Shape mismatch: expected {expected_shape}, got {list(tensor.shape)}"
            )
        
        # Validate dtype
        expected_dtype = torch.uint8 if not expected_config['normalize'] else torch.float32
        if tensor.dtype != expected_dtype:
            result['errors'].append(
                f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
            )
        
        # Validate value range
        if expected_config['normalize']:
            if tensor.min() < 0 or tensor.max() > 1:
                result['errors'].append(
                    f"Value range error: expected [0,1], got [{tensor.min():.4f}, {tensor.max():.4f}]"
                )
        else:
            if tensor.min() < 0 or tensor.max() > 255:
                result['errors'].append(
                    f"Value range error: expected [0,255], got [{tensor.min()}, {tensor.max()}]"
                )
        
        # Validate format
        expected_format = expected_config['format_type']
        actual_format = metadata.get('format', 'unknown')
        
        if actual_format != expected_format:
            result['errors'].append(
                f"Format mismatch: expected {expected_format}, got {actual_format}"
            )
        
        # Check for all-zero frames (potential corruption)
        for t in range(tensor.shape[0]):
            if tensor[t].sum() == 0:
                result['errors'].append(f"Frame {t} is all zeros (corrupted?)")
        
        # If no errors, mark as valid
        if not result['errors']:
            result['valid'] = True
            result['metadata'] = metadata
        
    except Exception as e:
        result['errors'].append(f"Exception during validation: {str(e)}")
    
    return result


def collect_cache_files() -> list:
    
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
    print(f"{'CACHE VALIDATION (PHASE 3)':^80}")
    print("="*80)
    print(f"  Cache root: {CACHE_ROOT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Load cache configuration
    config_path = CACHE_ROOT / "cache_config.json"
    
    if not config_path.exists():
        logger.error(f"Cache config not found: {config_path}")
        logger.error("Please run cache_frames_to_pt.py first.")
        return 1
    
    config = CacheConfig.load(config_path)
    logger.info(f"Loaded cache configuration from {config_path}")
    
    # Collect cache files
    logger.info("Collecting cache files...")
    cache_files = collect_cache_files()
    
    if not cache_files:
        logger.error("No cache files found!")
        return 1
    
    logger.info(f"Found {len(cache_files)} cache files to validate")
    
    # Validate in parallel
    results = []
    stats = defaultdict(int)
    
    expected_config = {
        'num_frames': config.num_frames,
        'resolution': config.resolution,
        'normalize': config.normalize,
        'format_type': config.format_type
    }
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(validate_single_cache, cf, expected_config): cf
            for cf in cache_files
        }
        
        with tqdm(
            total=len(cache_files),
            desc="Validating cache",
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
        'configuration': expected_config,
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
            for item in invalid:
                f.write(f"{item['file']}\n")
                for error in item['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
    
    # Print summary
    print("\n" + "="*80)
    print(f"{'VALIDATION SUMMARY':^80}")
    print("="*80)
    print(f"  Total cache files: {len(results):,}")
    print(f"   Valid: {stats['valid']:,} ({report['valid_percentage']:.1f}%)")
    print(f"   Invalid: {stats['invalid']:,}")
    print("="*80)
    
    if stats['invalid'] > 0:
        print(f"\n  {stats['invalid']} cache files have issues!")
        print(f"   Review: {INVALID_FILES}")
        print("   Consider re-caching invalid videos.")
    else:
        print("\n All cache files are valid and ready for training!")
    
    print(f"\n Full report: {VALIDATION_REPORT}")
    print("="*80 + "\n")
    
    return 0 if stats['invalid'] == 0 else 1


if __name__ == "__main__":
    exit(main())