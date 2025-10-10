import sys
from pathlib import Path

# Add project root to path BEFORE other imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now import project modules
from utils.setup_env import setup_environment  # noqa: E402
import tensorflow as tf  # noqa: E402


if __name__ == "__main__":
    config = setup_environment()
    x = tf.constant([1.0, 2.0])
    print("Tensor dtype:", x.dtype)
    print("Config:", config)
