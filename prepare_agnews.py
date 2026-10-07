"""Prepare a stratified calibration split from AG News training data."""
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets, load_dataset

OUT = Path('data/ag-news'); OUT.mkdir(parents=True, exist_ok=True)
source = load_dataset('fancyzhx/ag_news')
train = source['train']
labels = sorted(set(train['label']))
calibration_per_class = 200
calibration_parts = []
remaining_parts = []
for label in labels:
    group = train.filter(lambda row, wanted=label: row['label'] == wanted).shuffle(seed=42)
    calibration_parts.append(group.select(range(calibration_per_class)))
    remaining_parts.append(group.select(range(calibration_per_class, len(group))))

calibration = concatenate_datasets(calibration_parts).shuffle(seed=42)
train_remaining = concatenate_datasets(remaining_parts).shuffle(seed=42)
prepared = DatasetDict({'calibration': calibration, 'train_remaining': train_remaining, 'test': source['test']})
prepared.save_to_disk(str(OUT))
print(f'Calibration: {len(calibration)}; remaining train: {len(train_remaining)}; test: {len(source["test"])}; classes: {labels}')
