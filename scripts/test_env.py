import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.path_setup import add_project_root
from utils.setup_env import setup_environment
import tensorflow as tf

if __name__ == "__main__":
    config = setup_environment()
    x = tf.constant([1.0, 2.0])
    print("Tensor dtype:", x.dtype)
    print("Config:", config)
