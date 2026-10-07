"""Download deepset/prompt-injections and make a deterministic 80/20 split."""
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 42
OUT = Path('data'); OUT.mkdir(exist_ok=True)
dataset = load_dataset('deepset/prompt-injections')
split = concatenate_datasets(list(dataset.values())) if len(dataset) > 1 else next(iter(dataset.values()))
# Keep only examples whose dominant language detector result is English.
before = len(split)
def is_english(row):
    try: return detect(row['text']) == 'en'
    except LangDetectException: return False
split = split.filter(is_english)
print(f'English-only filter: {before} -> {len(split)} rows')
# Stratify manually because Dataset.train_test_split does not support a label column
# consistently across all datasets versions.
groups = []
for label in sorted(set(split['label'])):
    group = split.filter(lambda row, wanted=label: row['label'] == wanted)
    groups.append(group.shuffle(seed=42).train_test_split(test_size=0.2, seed=42))
parts = DatasetDict({
    'train': concatenate_datasets([group['train'] for group in groups]).shuffle(seed=42),
    'test': concatenate_datasets([group['test'] for group in groups]).shuffle(seed=42),
    'all_english': split.shuffle(seed=42),
})
parts.save_to_disk(str(OUT / 'prompt-injections'))
print(f"Saved {len(parts['train'])} train / {len(parts['test'])} test rows to {OUT / 'prompt-injections'}")
