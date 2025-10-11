import os, torch, numpy as np

class CacheManager:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save(self, name, tensor):
        path = os.path.join(self.cache_dir, f"{name}.pt")
        torch.save(tensor, path)

    def load(self, name):
        path = os.path.join(self.cache_dir, f"{name}.pt")
        return torch.load(path) if os.path.exists(path) else None
