"""Download deepset/prompt-injections and make a deterministic 80/20 split."""
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
OUT = Path('data'); OUT.mkdir(exist_ok=True)
dataset = load_dataset('deepset/prompt-injections')
split = concatenate_datasets(list(dataset.values())) if len(dataset) > 1 else next(iter(dataset.values()))
parts = split.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
parts.save_to_disk(str(OUT / 'prompt-injections'))
print(f"Saved {len(parts['train'])} train / {len(parts['test'])} test rows to {OUT / 'prompt-injections'}")
