import tensorflow as tf


print("TensorFlow version:", tf.__version__)
print("\n" + "=" * 60)
print("GPU Detection:")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")
print(f"Number of GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")

        gpu_details = tf.config.experimental.get_device_details(gpu)
        print(f"  Details: {gpu_details}")

    print("\n" + "=" * 60)
    print("Compute Capability Check:")
    print("=" * 60)
    print("For optimal mixed_float16 performance, you need:")
    print("  - NVIDIA GPU with compute capability ≥ 7.0")
    print("  - Examples: RTX 20/30/40 series, V100, A100, T4")

    print("\n" + "=" * 60)
    print("Testing GPU:")
    print("=" * 60)
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(" Matrix multiplication on GPU successful!")
        print(f"  Result:\n{c.numpy()}")
else:
    print("\n No GPU detected!")
    print("\nPossible reasons:")
    print("  1. CUDA/cuDNN not installed")
    print("  2. GPU drivers not installed")
    print("  3. TensorFlow CPU-only version installed")
    print("  4. GPU not compatible")

    print("\nCUDA available:", tf.test.is_built_with_cuda())
