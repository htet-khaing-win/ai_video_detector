import os

# Path to directory
base_dir = r"D:\GenBuster200k\processed\frames\train"

# List to store corrupted folders
corrupted = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        if len(files) < 16:
            corrupted.append(folder_name)
if corrupted:
    print("Corrupted folders:")
    for name in corrupted:
        print(f"- {name}")
else:
    print(" No corrupted folders found")
