# scripts/test_env.py

import sys
from pathlib import Path

# Add project root to path BEFORE other imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from utils.setup_env import setup_environment  # noqa: E402

if __name__ == "__main__":
    config = setup_environment()
    print("Config:", config)
