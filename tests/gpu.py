import os
import sys

print("=== Adding MKL DLL directories to PATH ===\n")

# Add Library\bin to PATH (where MKL DLLs are)
library_bin = os.path.join(sys.prefix, 'Library', 'bin')
torch_lib = os.path.join(sys.prefix, 'lib', 'site-packages', 'torch', 'lib')
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"

# Add all paths
os.environ['PATH'] = library_bin + os.pathsep + torch_lib + os.pathsep + cuda_bin + os.pathsep + os.environ.get('PATH', '')

# For Python 3.8+, also use add_dll_directory
if sys.version_info >= (3, 8):
    if os.path.exists(library_bin):
        os.add_dll_directory(library_bin)
        print(f"✓ Added to DLL search: {library_bin}")
    
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        print(f"✓ Added to DLL search: {torch_lib}")
    
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)
        print(f"✓ Added to DLL search: {cuda_bin}")

print("\n" + "="*60)
print("\n=== Attempting to import PyTorch ===\n")

try:
    import torch
    print("✓✓✓ SUCCESS! PyTorch imported successfully! ✓✓✓")
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print("\n🎉 Your GPU is detected and ready to use! 🎉")
    else:
        print("\n⚠ CUDA not available, but PyTorch works on CPU")
        
except Exception as e:
    print(f"✗ Failed to import PyTorch")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)