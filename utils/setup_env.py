import os
import random
import numpy as np
import tensorflow as tf
import logging
import yaml

def setup_environment(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    mixed_precision = config.get("mixed_precision", True)
    log_level = config.get("log_level", "INFO")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Mixed precision
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logging.info(f"Environment set. Seed={seed}, Mixed precision={mixed_precision}")
    return config
