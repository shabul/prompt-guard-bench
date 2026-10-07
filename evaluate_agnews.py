"""Evaluate a frozen Gemma 3 1B encoder with a four-class ridge head."""
import hashlib, json, math, time
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/Users/Shabul/model-weights/Google/gemma-3-1b-it'
DATA_PATH = Path('data/ag-news')
if not DATA_PATH.exists():
    raise FileNotFoundError('Run prepare_agnews.py first')
DATA = load_from_disk(str(DATA_PATH))
calibration, test = DATA['calibration'], DATA['test']
assert len(calibration) == 800 and len(test) == 7600
LABELS = ['World', 'Sports', 'Business', 'Science/Technology']
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(DEVICE).eval()

def embed_batch(texts):
    rendered = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True) for text in texts]
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(rendered, return_tensors='pt', padding=True, truncation=True, max_length=2048).items()}
    started = time.perf_counter()
    with torch.inference_mode(): outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if DEVICE == 'mps': torch.mps.synchronize()
    last_positions = inputs['attention_mask'].sum(dim=1) - 1
    vectors = outputs.hidden_states[-1][torch.arange(len(texts), device=DEVICE), last_positions].float().cpu()
    elapsed = (time.perf_counter() - started) * 1000
    return vectors, elapsed

def collect(dataset):
    rows = []
    batch_size = 64
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        vectors, elapsed = embed_batch(batch['text'])
        per_row_latency = elapsed / len(batch)
        rows.extend({'x': vector, 'label': int(label), 'latency_ms': per_row_latency} for vector, label in zip(vectors, batch['label']))
    return rows

cal, test_rows = collect(calibration), collect(test)
X = torch.stack([row['x'] for row in cal]); y = torch.tensor([row['label'] for row in cal], dtype=torch.long)
mean, scale = X.mean(0), X.std(0).clamp_min(1e-6); X = (X - mean) / scale
X = torch.cat([X, torch.ones(len(X), 1)], dim=1)
Y = torch.nn.functional.one_hot(y, num_classes=len(LABELS)).float()
regularizer = torch.eye(X.shape[1]); regularizer[-1, -1] = 0
weights = torch.linalg.solve(X.T @ X + regularizer, X.T @ Y)

def score(rows):
    vectors = torch.stack([row['x'] for row in rows]); vectors = (vectors - mean) / scale
    vectors = torch.cat([vectors, torch.ones(len(vectors), 1)], dim=1)
    probabilities = torch.softmax(vectors @ weights, dim=1)
    predictions = probabilities.argmax(dim=1).tolist()
    return [dict(row, prediction=prediction, probabilities=probability.tolist(), confidence=float(probability.max())) for row, prediction, probability in zip(rows, predictions, probabilities)]

scored = score(test_rows)
true = [row['label'] for row in scored]; pred = [row['prediction'] for row in scored]
confusion = [[sum(actual == i and predicted == j for actual, predicted in zip(true, pred)) for j in range(len(LABELS))] for i in range(len(LABELS))]
per_class = {}
for label, name in enumerate(LABELS):
    tp = confusion[label][label]; fp = sum(confusion[i][label] for i in range(len(LABELS)) if i != label); fn = sum(confusion[label][j] for j in range(len(LABELS)) if j != label)
    precision = tp / (tp + fp) if tp + fp else 0; recall = tp / (tp + fn) if tp + fn else 0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(confusion[label])}
accuracy = sum(actual == predicted for actual, predicted in zip(true, pred)) / len(true)
macro = {metric: sum(row[metric] for row in per_class.values()) / len(LABELS) for metric in ('precision', 'recall', 'f1')}
top3 = sum(actual in torch.topk(torch.tensor(row['probabilities']), 3).indices.tolist() for actual, row in zip(true, scored)) / len(true)
summary = {'model': MODEL, 'method': 'frozen final hidden state + four-class closed-form ridge head', 'device': DEVICE, 'dtype': 'float32', 'labels': LABELS, 'calibration_rows': len(calibration), 'test_rows': len(test), 'accuracy': accuracy, 'macro': macro, 'top3_accuracy': top3, 'per_class': per_class, 'confusion_matrix': confusion, 'mean_latency_ms': sum(row['latency_ms'] for row in scored) / len(scored), 'p95_latency_ms': sorted(row['latency_ms'] for row in scored)[int(len(scored) * .95) - 1], 'dataset_sha256': hashlib.sha256(''.join(row['text'] for row in test).encode()).hexdigest()}
Path('agnews_results.json').write_text(json.dumps({'summary': summary}, indent=2)); print(json.dumps(summary, indent=2))
