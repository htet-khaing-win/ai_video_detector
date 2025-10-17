import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING CROSS-GENERATOR SPLITS")
print("="*60)

# Load data
train_df = pd.read_csv('data/splits/train_metadata.csv')
test_df = pd.read_csv('data/splits/test_metadata.csv')

# Combine for re-splitting
all_df = pd.concat([train_df, test_df], ignore_index=True)

print(f"\nTotal samples: {len(all_df)}")
print(f"Label distribution: {dict(all_df['label'].value_counts())}")
print(f"\nGenerators: {all_df['generator'].unique()}")

# Strategy: Hold out ONE generator for validation
held_out_gen = 'ltxvideo'

print(f"\n Creating splits with '{held_out_gen}' held out...")

# Training set: All real + 3 fake generators (from TRAIN split only to preserve cache paths)
train_real = train_df[train_df['generator'] == 'real'].copy()
train_fake = train_df[train_df['generator'].isin(['cogvideox', 'easyanimate', 'hunyuanvideo'])].copy()

# Validation set: Real from TEST split + held-out generator
val_real = test_df[test_df['generator'] == 'real'].copy()
val_fake_from_train = train_df[train_df['generator'] == held_out_gen].copy()
val_fake_from_test = test_df[test_df['generator'] == held_out_gen].copy()

# Combine training
train_new = pd.concat([train_real, train_fake], ignore_index=True)

# Combine validation
val_new = pd.concat([val_real, val_fake_from_train, val_fake_from_test], ignore_index=True)


print(f"\n New Split Statistics:")
print(f"\nTrain: {len(train_new)} samples")
print(f"  Real: {(train_new['label'] == 0).sum()}")
print(f"  Fake: {(train_new['label'] == 1).sum()}")
print(f"  Generators: {train_new['generator'].value_counts().to_dict()}")

print(f"\nValidation: {len(val_new)} samples")
print(f"  Real: {(val_new['label'] == 0).sum()}")
print(f"  Fake: {(val_new['label'] == 1).sum()}")
print(f"  Generators: {val_new['generator'].value_counts().to_dict()}")

# Verify cache paths exist
print(f"\n Verifying cache files...")
missing_count = 0
for idx, row in train_new.iterrows():
    split_name = row['split']
    vid_id = row['video_id']
    cache_path = Path(f"data/processed/genbuster_cached/{split_name}/{vid_id}.pt")
    if not cache_path.exists():
        missing_count += 1
        if missing_count <= 3:
            print(f"   Missing: {cache_path}")

if missing_count > 0:
    print(f"\n WARNING: {missing_count} cache files missing in TRAIN!")
else:
    print(f"   All train cache files exist")

# Check val cache
missing_val = 0
for idx, row in val_new.iterrows():
    split_name = row['split']
    vid_id = row['video_id']
    cache_path = Path(f"data/processed/genbuster_cached/{split_name}/{vid_id}.pt")
    if not cache_path.exists():
        missing_val += 1
        if missing_val <= 3:
            print(f"   Missing: {cache_path}")

if missing_val > 0:
    print(f"\n WARNING: {missing_val} cache files missing in VAL!")
else:
    print(f"   All val cache files exist")

# Save new splits
train_new.to_csv('data/splits/train_crossgen.csv', index=False)
val_new.to_csv('data/splits/val_crossgen.csv', index=False)

print(f"\n Saved:")
print(f"  data/splits/train_crossgen.csv")
print(f"  data/splits/val_crossgen.csv")

print("\n" + "="*60)
print("CROSS-GENERATOR SPLIT EXPLANATION:")
print("="*60)
