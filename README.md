# ai_video_detector

# Setup Instructions

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.1 support (for GPU acceleration)
- Windows: Visual C++ Redistributables (2015-2022)
- NVIDIA drivers (latest)

## Installation

### Linux / CI (GitHub Actions)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Windows

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt

# Install Windows-specific MKL libraries (REQUIRED)
pip install -r requirements-windows.txt

# Run Windows setup script to fix DLL paths
python setup_windows.py

# For development
pip install -r requirements-dev.txt
```

## Verification

Test that PyTorch and CUDA are working:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.4.0+cu121
CUDA available: True
```

## Troubleshooting

### Windows: DLL Error (WinError 126)

If you get DLL errors on Windows:

1. Install Visual C++ Redistributables:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart

2. Install Windows-specific requirements:
   ```bash
   pip install -r requirements-windows.txt
   ```

3. Run the setup script:
   ```bash
   python setup_windows.py
   ```

### Linux: No GPU detected

- Update NVIDIA drivers: `sudo ubuntu-drivers autoinstall`
- Verify CUDA: `nvidia-smi`

### CI/CD: Tests failing

- Ensure `requirements.txt` doesn't include Windows-specific packages (mkl)
- Use `requirements-dev.txt` in CI workflows
