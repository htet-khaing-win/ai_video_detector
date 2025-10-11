import os
import pytest
from src.data.download_datasets import stratified_sample

def test_stratified_sample_small():
    # small synthetic dataset
    examples = [
        {"label": 0, "generator": "A"},
        {"label": 0, "generator": "B"},
        {"label": 1, "generator": "A"},
        {"label": 1, "generator": "B"},
    ]
    selected = stratified_sample(examples, label_col="label", strat_cols=["generator"], target=4, seed=42)
    assert len(selected) == 4
    assert set(selected) <= set(range(4))
