import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("CREATING CROSS-GENERATOR SPLITS")
print("="*60)

# Load your full data
train_df = pd.read_csv('data/splits/train_metadata.csv')
test_df = pd.read_csv('data/splits/test_metadata.csv')

# Combine for re-splitting
all_df = pd.concat([train_df, test_df], ignore_index=True)

print(f"\nTotal samples: {len(all_df)}")
print(f"Label distribution: {dict(all_df['label'].value_counts())}")
print(f"\nGenerators: {all_df['generator'].unique()}")

# Strategy: Hold out ONE generator for validation
held_out_gen = 'ltxvideo'  # Smallest fake generator

print(f"\n Creating splits with '{held_out_gen}' held out...")

# Training set: All real + 3 fake generators
train_mask = (all_df['generator'] == 'real') | \
             (all_df['generator'].isin(['cogvideox', 'easyanimate', 'hunyuanvideo']))

train_new = all_df[train_mask].copy()

# Validation set: Some real + held-out generator
val_real = all_df[all_df['generator'] == 'real'].sample(n=1000, random_state=42)
val_fake = all_df[all_df['generator'] == held_out_gen].copy()

# Remove val_real samples from train
train_new = train_new[~train_new['video_id'].isin(val_real['video_id'])]

val_new = pd.concat([val_real, val_fake], ignore_index=True)

print(f"\n New Split Statistics:")
print(f"\nTrain: {len(train_new)} samples")
print(f"  Real: {(train_new['label'] == 0).sum()}")
print(f"  Fake: {(train_new['label'] == 1).sum()}")
print(f"  Generators: {train_new['generator'].value_counts().to_dict()}")

print(f"\nValidation: {len(val_new)} samples")
print(f"  Real: {(val_new['label'] == 0).sum()}")
print(f"  Fake: {(val_new['label'] == 1).sum()}")
print(f"  Generators: {val_new['generator'].value_counts().to_dict()}")

# Save new splits
train_new.to_csv('data/splits/train_crossgen.csv', index=False)
val_new.to_csv('data/splits/val_crossgen.csv', index=False)

print(f"\n Saved:")
print(f"  data/splits/train_crossgen.csv")
print(f"  data/splits/val_crossgen.csv")

print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
