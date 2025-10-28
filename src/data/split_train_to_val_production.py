"""
Production-grade train/validation split script for MLOps pipeline.

Features:
- Stratified split maintaining label distribution
- Reproducible with fixed random seed
- Preserves original CSV schema and column order
- UTF-8 encoding, forward slashes in paths
- Comprehensive logging and statistics
- Error handling and validation

Input:  C:/Personal project/ai_video_detector/data/splits/train_metadata.csv
Output: C:/Personal project/ai_video_detector/data/splits/train_split.csv (80%)
        C:/Personal project/ai_video_detector/data/splits/val_split.csv (20%)
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/split_train_val.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import random seed from utils (if available)
try:
    project_root = Path("C:/Personal project/ai_video_detector")
    sys.path.insert(0, str(project_root))
    from utils.setup_env import get_random_seed
    RANDOM_SEED = get_random_seed()
    logger.info(f"Using random seed from utils.setup_env: {RANDOM_SEED}")
except ImportError:
    RANDOM_SEED = 42
    logger.warning("utils.setup_env not found. Using default seed: 42")


def validate_csv_schema(df, required_columns):
    """
    Validate that CSV has all required columns.
    
    Args:
        df: pandas DataFrame
        required_columns: list of required column names
    
    Returns:
        bool: True if valid, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    return True


def print_split_statistics(df, split_name):
    """
    Print detailed statistics about a split.
    
    Args:
        df: pandas DataFrame
        split_name: name of the split (e.g., "Train", "Validation")
    """
    total = len(df)
    real_count = (df['label'] == 0).sum()
    fake_count = (df['label'] == 1).sum()
    real_pct = 100 * real_count / total if total > 0 else 0
    fake_pct = 100 * fake_count / total if total > 0 else 0
    
    logger.info(f"\n{split_name} Split Statistics:")
    logger.info(f"  Total videos: {total:,}")
    logger.info(f"  Real videos:  {real_count:,} ({real_pct:.2f}%)")
    logger.info(f"  Fake videos:  {fake_count:,} ({fake_pct:.2f}%)")
    
    # Generator breakdown (if available)
    if 'generator' in df.columns:
        logger.info(f"  Generator breakdown:")
        gen_counts = df['generator'].value_counts()
        for gen, count in gen_counts.items():
            logger.info(f"    {gen:20s}: {count:,}")


def split_train_validation(
    train_csv="C:/Personal project/ai_video_detector/data/splits/train_metadata.csv",
    output_dir="C:/Personal project/ai_video_detector/data/splits",
    val_ratio=0.2,
    random_state=None
):
    """
    Split train dataset into train + validation with stratification.
    
    Args:
        train_csv: path to original train metadata CSV
        output_dir: directory to save output CSVs
        val_ratio: fraction of data for validation (default: 0.2 = 20%)
        random_state: random seed for reproducibility (uses RANDOM_SEED if None)
    
    Returns:
        tuple: (train_df, val_df) if successful, (None, None) if failed
    """
    if random_state is None:
        random_state = RANDOM_SEED
    
    logger.info("="*70)
    logger.info("Train/Validation Split Pipeline")
    logger.info("="*70)
    logger.info(f"Input CSV:     {train_csv}")
    logger.info(f"Output dir:    {output_dir}")
    logger.info(f"Val ratio:     {val_ratio}")
    logger.info(f"Random seed:   {random_state}")
    logger.info(f"Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate paths
    train_csv = Path(train_csv)
    output_dir = Path(output_dir)
    
    if not train_csv.exists():
        logger.error(f"Train CSV not found: {train_csv}")
        return None, None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\nLoading train metadata...")
    try:
        df = pd.read_csv(train_csv, encoding='utf-8')
        logger.info(f" Loaded {len(df):,} videos")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None, None
    
    # Validate schema
    required_columns = ['video_id', 'filepath', 'absolute_path', 'label', 'split', 'dataset', 'generator']
    if not validate_csv_schema(df, required_columns):
        logger.error("CSV schema validation failed")
        return None, None
    
    # Print original statistics
    print_split_statistics(df, "Original Train")
    
    # Check for missing labels
    if df['label'].isna().any():
        logger.warning(f"Found {df['label'].isna().sum()} videos with missing labels")
        df = df.dropna(subset=['label'])
        logger.info(f"Dropped missing labels. Remaining: {len(df):,}")
    
    # Perform stratified split
    logger.info(f"\nPerforming stratified split (train: {1-val_ratio:.0%}, val: {val_ratio:.0%})...")
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=df['label'],  # Stratify by label to maintain real/fake ratio
            shuffle=True
        )
        logger.info(" Split completed successfully")
    except Exception as e:
        logger.error(f"Split failed: {e}")
        return None, None
    
    # Update split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # Print statistics for each split
    print_split_statistics(train_df, "Train")
    print_split_statistics(val_df, "Validation")
    
    # Verify stratification worked
    train_real_pct = 100 * (train_df['label'] == 0).sum() / len(train_df)
    val_real_pct = 100 * (val_df['label'] == 0).sum() / len(val_df)
    pct_diff = abs(train_real_pct - val_real_pct)
    
    if pct_diff > 1.0:  # Allow 1% difference
        logger.warning(f"Stratification check: {pct_diff:.2f}% difference in real/fake ratio")
    else:
        logger.info(f" Stratification verified (difference: {pct_diff:.3f}%)")
    
    # Save CSVs
    train_output = output_dir / "train_split.csv"
    val_output = output_dir / "val_split.csv"
    
    logger.info("\nSaving split CSVs...")
    try:
        # Preserve column order and formatting
        train_df.to_csv(train_output, index=False, encoding='utf-8')
        val_df.to_csv(val_output, index=False, encoding='utf-8')
        
        logger.info(f" Train split saved: {train_output}")
        logger.info(f" Val split saved:   {val_output}")
    except Exception as e:
        logger.error(f"Failed to save CSVs: {e}")
        return None, None
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info(" Split Pipeline Complete!")
    logger.info("="*70)
    logger.info(f"Train set: {len(train_df):,} videos ({train_output.name})")
    logger.info(f"Val set:   {len(val_df):,} videos ({val_output.name})")
    logger.info(f"Total:     {len(train_df) + len(val_df):,} videos")
    
    return train_df, val_df


def verify_splits(train_csv, val_csv):
    """
    Verify that splits don't have overlapping videos.
    
    Args:
        train_csv: path to train_split.csv
        val_csv: path to val_split.csv
    
    Returns:
        bool: True if no overlap, False if overlap detected
    """
    logger.info("\nVerifying split integrity...")
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    train_ids = set(train_df['video_id'])
    val_ids = set(val_df['video_id'])
    
    overlap = train_ids & val_ids
    
    if overlap:
        logger.error(f" Found {len(overlap)} overlapping videos between train and val!")
        logger.error(f"Examples: {list(overlap)[:5]}")
        return False
    else:
        logger.info(f" No overlap detected. Splits are independent.")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split train metadata into train and validation sets"
    )
    parser.add_argument(
        '--train_csv',
        default='C:/Personal project/ai_video_detector/data/splits/train_metadata.csv',
        help='Path to input train metadata CSV'
    )
    parser.add_argument(
        '--output_dir',
        default='C:/Personal project/ai_video_detector/data/splits',
        help='Output directory for split CSVs'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: from utils.setup_env or 42)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify splits after creation'
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Run split
    train_df, val_df = split_train_validation(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        random_state=args.seed
    )
    
    # Verify if requested
    if args.verify and train_df is not None and val_df is not None:
        train_output = Path(args.output_dir) / "train_split.csv"
        val_output = Path(args.output_dir) / "val_split.csv"
        verify_splits(train_output, val_output)