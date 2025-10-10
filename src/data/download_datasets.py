"""
Download and stratified-sample GenBuster-200K-mini from Hugging Face.

Usage:
  python src/data/download_datasets.py --out_dir data/raw/genbuster-200k-mini --target 50000 --seed 42
"""

import argparse
import os
import random
import csv
import glob
import shutil
from collections import defaultdict
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import py7zr
from tqdm import tqdm


def download_and_extract_7z(repo_id, filename, output_dir, token=True, cleanup=True):
    """Download a .7z file from HuggingFace and extract it"""
    
    print(f"Downloading {filename}...")
    
    # Download the .7z file
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        cache_dir="./cache"
    )
    
    print(f"Extracting {filename} to {output_dir}...")
    
    # Extract the .7z file
    os.makedirs(output_dir, exist_ok=True)
    
    with py7zr.SevenZipFile(downloaded_path, mode='r') as archive:
        archive.extractall(path=output_dir)
    
    print(f"Extracted {filename}")
    
    # Optional: Delete the .7z file after extraction to save space
    if cleanup:
        try:
            os.remove(downloaded_path)
            print(f"Cleaned up {filename} (saved disk space)")
        except Exception as e:
            print(f"Could not delete {filename}: {e}")
    
    return output_dir


def download_7z_files(repo_id, out_dir, token=True):
    """Download and extract all .7z files from the repository"""
    
    print(f"Listing files in {repo_id}...")
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        
        # Filter for .7z files
        seven_z_files = [f for f in files if f.endswith('.7z')]
        
        if not seven_z_files:
            print("No .7z files found in repository")
            return False
        
        print(f"Found {len(seven_z_files)} .7z files:")
        for f in seven_z_files:
            print(f"  {f}")
        
        # Download and extract each .7z file
        for seven_z_file in seven_z_files:
            download_and_extract_7z(
                repo_id=repo_id,
                filename=seven_z_file,
                output_dir=out_dir,
                token=token,
                cleanup=True
            )
        
        print(f"All files extracted to: {out_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading .7z files: {e}")
        import traceback
        traceback.print_exc()
        return False


def stratified_sample(
    dataset, label_col="label", strat_cols=None, target=50000, seed=42
):
    # strat_cols: list of columns to stratify by in addition to label
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, ex in enumerate(dataset):
        key_parts = [str(ex.get(label_col, "unknown"))]
        if strat_cols:
            for c in strat_cols:
                key_parts.append(str(ex.get(c, "NA")))
        key = "|".join(key_parts)
        groups[key].append(i)
    # compute per-label target proportional to group sizes but ensure label balance
    # first split by label
    label_to_indices = defaultdict(list)
    for key, inds in groups.items():
        label = key.split("|")[0]
        label_to_indices[label].extend(inds)
    # enforce exact balance across labels if possible
    labels = list(label_to_indices.keys())
    per_label = target // len(labels)
    selected = []
    for lab in labels:
        inds = label_to_indices[lab]
        rng.shuffle(inds)
        selected.extend(inds[:per_label])
    return selected


def main(out_dir, target, seed):
    os.makedirs(out_dir, exist_ok=True)
    
    repo_id = "l8cv/GenBuster-200K-mini"
    
    print("Checking for .7z files in repository...")
    has_7z = download_7z_files(repo_id, out_dir, token=True)
    
    if has_7z:
        print("Dataset downloaded and extracted from .7z files")
        # Count extracted files
        extracted_files = glob.glob(os.path.join(out_dir, "**/*"), recursive=True)
        extracted_files = [f for f in extracted_files if os.path.isfile(f)]
        print(f"Total extracted files: {len(extracted_files)}")
        
        # If target specified, create subset
        if target and len(extracted_files) > target:
            print(f"Creating subset of {target} samples...")
            rng = random.Random(seed)
            selected_files = rng.sample(extracted_files, target)
            
            # Move selected files to subset directory
            subset_dir = f"{out_dir}_subset_{target}"
            os.makedirs(subset_dir, exist_ok=True)
            
            for file_path in tqdm(selected_files, desc="Copying subset"):
                relative_path = os.path.relpath(file_path, out_dir)
                target_path = os.path.join(subset_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(file_path, target_path)
            
            print(f"Subset saved to: {subset_dir}")
        
        return
    
    # Fall back to original dataset loading method
    print("Loading GenBuster-200K-mini from Hugging Face (this loads metadata).")
    try:
        ds = load_dataset(repo_id, split="train", token=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This dataset have a different structure.")
        return
    
    # Inspect columns to choose stratification fields safely
    print("Columns:", ds.column_names)
    strat_cols = []
    if "generator" in ds.column_names:
        strat_cols = ["generator"]
    selected_indices = stratified_sample(
        ds, label_col="label", strat_cols=strat_cols, target=target, seed=seed
    )
    print(f"Selected {len(selected_indices)} indices for sampled subset.")
    # Write metadata CSV for selected subset
    out_csv = os.path.join(out_dir, "genbuster_sampled_metadata.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        # header
        header = ds.column_names
        writer.writerow(header)
        for idx in selected_indices:
            row = [ds[int(idx)][c] for c in header]
            writer.writerow(row)
    print(f"Wrote sampled metadata to {out_csv}")
    print(
        "Note: actual video assets may not be provided directly via dataset. If asset URLs are present, use those to download or place videos under data/raw/genbuster-200k-mini/assets/"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw/genbuster-200k-mini")
    p.add_argument("--target", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.out_dir, args.target, args.seed)