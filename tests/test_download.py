import os
from src.data.download_genbuster import stratified_sample
from datasets import Dataset

def test_stratified_sample_small():
    # create small synthetic dataset
    examples = [
        {"label": 0, "generator": "A"},
        {"label": 0, "generator": "B"},
        {"label": 1, "generator": "A"},
        {"label": 1, "generator": "B"},
    ]
    ds = Dataset.from_list(examples)
    idxs = stratified_sample(ds, label_col="label", strat_cols=["generator"], target=4, seed=42)
    assert len(idxs) == 4