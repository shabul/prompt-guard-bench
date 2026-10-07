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
calibration_groups = []; test_groups = []; oob_groups = []
for label in sorted(set(split['label'])):
    group = split.filter(lambda row, wanted=label: row['label'] == wanted).shuffle(seed=42)
    first = group.train_test_split(test_size=0.4, seed=42)
    rest = first['test'].train_test_split(test_size=0.5, seed=42)
    calibration_groups.append(first['train']); test_groups.append(rest['train']); oob_groups.append(rest['test'])
parts = DatasetDict({
    'calibration': concatenate_datasets(calibration_groups).shuffle(seed=42),
    'test': concatenate_datasets(test_groups).shuffle(seed=42),
    'oob': concatenate_datasets(oob_groups).shuffle(seed=42),
    'all_english': split.shuffle(seed=42),
})
parts.save_to_disk(str(OUT / 'prompt-injections'))
print(f"Saved {len(parts['calibration'])} calibration / {len(parts['test'])} test / {len(parts['oob'])} OOB rows to {OUT / 'prompt-injections'}")
