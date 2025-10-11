import torch

def vram_usage_bytes():
    if not torch.cuda.is_available():
        return 0, 0
    used = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    return used, total

def recommend_batch_size(current_batch, used_threshold_bytes=7 * 1024**3):
    used, total = vram_usage_bytes()
    if used == 0:
        return current_batch
    if used > used_threshold_bytes:
        # naive reduce: halve batch until under threshold
        new_batch = max(1, current_batch // 2)
        return new_batch
    return current_batch

def log_and_warn(threshold_bytes=7 * 1024**3):
    used, total = vram_usage_bytes()
    if used == 0:
        print("No CUDA device available.")
        return
    pct = used / total
    print(f"VRAM used: {used/1024**3:.2f}GB / {total/1024**3:.2f}GB ({pct*100:.1f}%)")
    if used > threshold_bytes:
        print("Warning: VRAM approaching threshold. Consider reducing batch size.")
