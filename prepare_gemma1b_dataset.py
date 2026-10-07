"""Prepare the English-dominant prompt-injection dataset for Gemma 1B."""
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

OUT = Path('data'); OUT.mkdir(exist_ok=True)
source = load_dataset('deepset/prompt-injections')
dataset = concatenate_datasets(list(source.values()))
def is_english(row):
    try: return detect(row['text']) == 'en'
    except Exception: return False
english = dataset.filter(is_english)
calibration = []; test = []; oob = []
for label in sorted(set(english['label'])):
    group = english.filter(lambda row, wanted=label: row['label'] == wanted).shuffle(seed=42)
    calibration_n = round(150 * len(group) / len(english))
    test_n = round(100 * len(group) / len(english))
    calibration.append(group.select(range(calibration_n)))
    test.append(group.select(range(calibration_n, calibration_n + test_n)))
    oob.append(group.select(range(calibration_n + test_n, len(group))))
prepared = DatasetDict({
    'calibration': concatenate_datasets(calibration).shuffle(seed=42),
    'test': concatenate_datasets(test).shuffle(seed=42),
    'oob': concatenate_datasets(oob).shuffle(seed=42),
    'all_english': english.shuffle(seed=42),
})
prepared.save_to_disk(str(OUT / 'gemma1b-prompt-injections'))
print(f'English rows: {len(english)}; calibration: {len(prepared["calibration"])}; test: {len(prepared["test"])}; OOB: {len(prepared["oob"])}')
