"""Crash-safe MPS evaluation of a frozen Gemma 1B multi-class AG News head."""
import hashlib, json, time
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/Users/Shabul/model-weights/Google/gemma-3-1b-it'
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data/ag-news'; CACHE = DATA_PATH / 'embedding-cache-v2'; CACHE.mkdir(parents=True, exist_ok=True)
DATA = load_from_disk(str(DATA_PATH)); calibration, test = DATA['calibration'], DATA['test']
assert len(calibration) == 8000 and len(test) == 7600
LABELS = ['World', 'Sports', 'Business', 'Science/Technology']; DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'; BATCH_SIZE = 8
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(DEVICE).eval()

def embed_batch(texts):
    rendered = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True) for text in texts]
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(rendered, return_tensors='pt', padding=True, truncation=True, max_length=2048).items()}
    started = time.perf_counter()
    with torch.inference_mode(): outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if DEVICE == 'mps': torch.mps.synchronize()
    last_positions = inputs['attention_mask'].sum(dim=1) - 1
    vectors = outputs.hidden_states[-1][torch.arange(len(texts), device=DEVICE), last_positions].float().cpu()
    return vectors, (time.perf_counter() - started) * 1000

def collect(dataset, split):
    rows = []
    for start in range(0, len(dataset), BATCH_SIZE):
        path = CACHE / f'{split}-{start:06d}.pt'
        if path.exists():
            saved = torch.load(path, weights_only=True); vectors, labels, latencies = saved['vectors'], saved['labels'].tolist(), saved['latencies']
        else:
            batch = dataset.select(range(start, min(start + BATCH_SIZE, len(dataset))))
            vectors, elapsed = embed_batch(batch['text']); labels = [int(label) for label in batch['label']]; latencies = [elapsed / len(batch)] * len(batch)
            torch.save({'vectors': vectors, 'labels': torch.tensor(labels), 'latencies': latencies}, path)
            print(f'{split}: saved {min(start + BATCH_SIZE, len(dataset))}/{len(dataset)}', flush=True)
        rows.extend({'x': vector, 'label': label, 'latency_ms': latency} for vector, label, latency in zip(vectors, labels, latencies))
    return rows

cal, test_rows = collect(calibration, 'calibration'), collect(test, 'test')
X = torch.stack([row['x'] for row in cal]); y = torch.tensor([row['label'] for row in cal], dtype=torch.long)
mean, scale = X.mean(0), X.std(0).clamp_min(1e-6); X = (X - mean) / scale; X = torch.cat([X, torch.ones(len(X), 1)], dim=1)
Y = torch.nn.functional.one_hot(y, num_classes=len(LABELS)).float()

def evaluate(lam):
    regularizer = torch.eye(X.shape[1]); regularizer[-1, -1] = 0; weights = torch.linalg.solve(X.T @ X + lam * regularizer, X.T @ Y)
    vectors = torch.stack([row['x'] for row in test_rows]); vectors = (vectors - mean) / scale; vectors = torch.cat([vectors, torch.ones(len(vectors), 1)], dim=1)
    probabilities = torch.softmax(vectors @ weights, dim=1); predictions = probabilities.argmax(dim=1).tolist(); true = [row['label'] for row in test_rows]
    confusion = [[sum(actual == i and predicted == j for actual, predicted in zip(true, predictions)) for j in range(4)] for i in range(4)]; per_class = {}
    for label, name in enumerate(LABELS):
        tp = confusion[label][label]; fp = sum(confusion[i][label] for i in range(4) if i != label); fn = sum(confusion[label][j] for j in range(4) if j != label)
        precision = tp / (tp + fp) if tp + fp else 0; recall = tp / (tp + fn) if tp + fn else 0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0; per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(confusion[label])}
    macro = {metric: sum(item[metric] for item in per_class.values()) / 4 for metric in ('precision', 'recall', 'f1')}; accuracy = sum(a == b for a, b in zip(true, predictions)) / len(true)
    top3 = sum(a in torch.topk(probability, 3).indices.tolist() for a, probability in zip(true, probabilities)) / len(true)
    return {'lambda': lam, 'accuracy': accuracy, 'macro': macro, 'top3_accuracy': top3, 'per_class': per_class, 'confusion_matrix': confusion}

results = [evaluate(lam) for lam in (0.01, 0.1, 1.0, 10.0, 100.0)]; best = max(results, key=lambda item: item['macro']['f1']); latencies = [row['latency_ms'] for row in test_rows]
summary = {'model': MODEL, 'method': 'frozen final hidden state + tuned four-class closed-form ridge head', 'device': DEVICE, 'dtype': 'float32', 'labels': LABELS, 'calibration_rows': len(calibration), 'calibration_per_class': 2000, 'test_rows': len(test), 'regularization_sweep': results, 'best': best, 'mean_latency_ms': sum(latencies) / len(latencies), 'p95_latency_ms': sorted(latencies)[int(len(latencies) * .95) - 1], 'dataset_sha256': hashlib.sha256(''.join(row['text'] for row in test).encode()).hexdigest()}
(ROOT / 'agnews_results.json').write_text(json.dumps({'summary': summary}, indent=2)); print(json.dumps(summary, indent=2))
