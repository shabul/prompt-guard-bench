"""Evaluate a frozen Gemma 3 1B hidden-state ridge classifier."""
import hashlib, json, math, time
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/Users/Shabul/model-weights/Google/gemma-3-1b-it'
DATA = load_from_disk('data/gemma1b-prompt-injections')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
calibration, test, oob = DATA['calibration'], DATA['test'], DATA['oob']
assert len(calibration) == 150 and len(test) == 100 and len(oob) == 101
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(DEVICE).eval()

def embed(text):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(rendered, return_tensors='pt', truncation=True, max_length=2048).items()}
    started = time.perf_counter()
    with torch.inference_mode(): outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if DEVICE == 'mps': torch.mps.synchronize()
    return outputs.hidden_states[-1][0, -1].float().cpu(), (time.perf_counter() - started) * 1000

def collect(dataset):
    rows = []
    for row in dataset:
        vector, latency = embed(row['text'])
        rows.append({'x': vector, 'label': int(row['label']), 'latency_ms': latency})
    return rows

cal = collect(calibration); test_rows = collect(test); oob_rows = collect(oob)
X = torch.stack([row['x'] for row in cal]); y = torch.tensor([row['label'] for row in cal], dtype=torch.float32)
mean, scale = X.mean(0), X.std(0).clamp_min(1e-6); X = (X - mean) / scale
X = torch.cat([X, torch.ones(len(X), 1)], dim=1)
regularizer = torch.eye(X.shape[1]); regularizer[-1, -1] = 0
weights = torch.linalg.solve(X.T @ X + regularizer, X.T @ y)

def score(rows):
    vectors = torch.stack([row['x'] for row in rows]); vectors = (vectors - mean) / scale
    vectors = torch.cat([vectors, torch.ones(len(vectors), 1)], dim=1)
    probabilities = torch.sigmoid(vectors @ weights).tolist()
    return [dict(row, probability=probability) for row, probability in zip(rows, probabilities)]
def metrics(rows, threshold):
    assert all(math.isfinite(row['probability']) for row in rows)
    tp = sum(row['label'] == 1 and row['probability'] >= threshold for row in rows)
    fp = sum(row['label'] == 0 and row['probability'] >= threshold for row in rows)
    fn = sum(row['label'] == 1 and row['probability'] < threshold for row in rows)
    tn = sum(row['label'] == 0 and row['probability'] < threshold for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0; recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {'accuracy': (tp + tn) / len(rows), 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}, 'mean_latency_ms': sum(row['latency_ms'] for row in rows) / len(rows)}

test_rows, oob_rows = score(test_rows), score(oob_rows)
summary = {'model': MODEL, 'method': 'frozen final hidden state + closed-form ridge head', 'device': DEVICE, 'dtype': 'float32', 'calibration_rows': 150, 'test_rows': 100, 'oob_rows': 101, 'thresholds': {str(t): {'test': metrics(test_rows, t), 'oob': metrics(oob_rows, t)} for t in (0.5, 0.6, 0.7, 0.8, 0.9)}, 'dataset_sha256': hashlib.sha256(''.join(row['text'] for split in (calibration, test, oob) for row in split).encode()).hexdigest()}
Path('gemma1b_results.json').write_text(json.dumps({'summary': summary}, indent=2)); print(json.dumps(summary, indent=2))
