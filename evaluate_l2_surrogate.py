"""AnyJev-style L2 surrogate: frozen Gemma hidden states + closed-form ridge head."""
import hashlib, json, math, time
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='/Users/Shabul/model-weights/Google/gemma-3-270m-it'; DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
sets=load_from_disk('data/prompt-injections'); calibration, test, oob = sets['calibration'], sets['test'], sets['oob']
assert len(calibration)==150 and len(calibration)+len(test)+len(oob)==len(sets['all_english'])
tokenizer=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).to(DEVICE).eval()

def embedding(text):
    messages=[{'role':'user','content':text}]
    rendered=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tokenizer,'apply_chat_template') else text
    inputs=tokenizer(rendered, return_tensors='pt', truncation=True, max_length=2048)
    inputs={k:v.to(DEVICE) for k,v in inputs.items()}
    started=time.perf_counter()
    with torch.inference_mode(): outputs=model(**inputs, output_hidden_states=True, use_cache=False)
    if DEVICE=='mps': torch.mps.synchronize()
    hidden=outputs.hidden_states[-1][0, -1].float().cpu()
    return hidden, (time.perf_counter()-started)*1000

def collect(dataset):
    rows=[]
    for row in dataset:
        vector, latency=embedding(row['text']); rows.append({'x':vector,'label':int(row['label']),'latency_ms':latency})
    return rows

cal=collect(calibration); test_rows=collect(test); oob_rows=collect(oob)
X=torch.stack([r['x'] for r in cal]); y=torch.tensor([r['label'] for r in cal],dtype=torch.float32)
mean=X.mean(0); scale=X.std(0).clamp_min(1e-6); X=(X-mean)/scale
Xb=torch.cat([X,torch.ones(len(X),1)],dim=1); ridge=1.0; reg=torch.eye(Xb.shape[1]); reg[-1,-1]=0
weights=torch.linalg.solve(Xb.T@Xb + ridge*reg, Xb.T@y)

def score(rows):
    vectors=torch.stack([r['x'] for r in rows]); vectors=(vectors-mean)/scale; xb=torch.cat([vectors,torch.ones(len(vectors),1)],dim=1)
    probabilities=torch.sigmoid(xb@weights).tolist(); return [dict(r,probability=p) for r,p in zip(rows,probabilities)]
def metrics(rows):
    for r in rows: assert math.isfinite(r['probability']) and 0<=r['probability']<=1
    tp=sum(r['label']==1 and r['probability']>=.5 for r in rows); fp=sum(r['label']==0 and r['probability']>=.5 for r in rows); fn=sum(r['label']==1 and r['probability']<.5 for r in rows); tn=sum(r['label']==0 and r['probability']<.5 for r in rows)
    p=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*p*recall/(p+recall) if p+recall else 0
    return {'n':len(rows),'accuracy':(tp+tn)/len(rows),'precision':p,'recall':recall,'f1':f1,'confusion_matrix':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},'mean_latency_ms':sum(r['latency_ms'] for r in rows)/len(rows)}
test_scored=score(test_rows); oob_scored=score(oob_rows)
def public(rows): return [{k:v for k,v in row.items() if k != 'x'} for row in rows]
summary={'model':MODEL,'method':'AnyJev-style L2 surrogate: frozen hidden state + closed-form ridge head','device':DEVICE,'dtype':'float32','ridge':ridge,'threshold':.5,'sizes':{'calibration':len(cal),'test':len(test_rows),'oob':len(oob_rows)},'calibration_labels':{'safe':sum(r['label']==0 for r in cal),'injection':sum(r['label']==1 for r in cal)},'test':metrics(test_scored),'oob':metrics(oob_scored),'dataset_sha256':hashlib.sha256(''.join(row['text'] for dataset in (calibration,test,oob) for row in dataset).encode()).hexdigest()}
Path('l2_surrogate_results.json').write_text(json.dumps({'summary':summary,'test_rows':public(test_scored),'oob_rows':public(oob_scored)},indent=2)); print(json.dumps(summary,indent=2))
