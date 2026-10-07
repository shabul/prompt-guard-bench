"""Evaluate the frozen Gemma 1B head on Giskard's positive-only CSV.

Because this CSV contains only Jailbreak/Hijacking attacks, this reports recall
and latency only. Precision, accuracy, and F1 are intentionally not computed.
"""
import csv, json, subprocess, sys, time, urllib.request
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='/Users/Shabul/model-weights/Google/gemma-3-1b-it'
URL='https://raw.githubusercontent.com/Giskard-AI/prompt-injections/main/prompt_injections.csv'
csv_path=Path('data/giskard_prompt_injections.csv'); csv_path.parent.mkdir(exist_ok=True)
if not csv_path.exists(): urllib.request.urlretrieve(URL, csv_path)
with csv_path.open(encoding='utf-8', newline='') as f: giskard=list(csv.DictReader(f))
assert len(giskard)==35 and all(row['group'] in {'Jailbreak','Hijacking attacks'} for row in giskard)
prepared_path=Path('data/gemma1b-prompt-injections')
if not prepared_path.exists():
    subprocess.run([sys.executable, 'prepare_gemma1b_dataset.py'], check=True)
DATA=load_from_disk(str(prepared_path)); calibration=DATA['calibration']
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,trust_remote_code=True).to(DEVICE).eval()
def embed(text):
    rendered=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    inputs={k:v.to(DEVICE) for k,v in tokenizer(rendered,return_tensors='pt',truncation=True,max_length=2048).items()}
    started=time.perf_counter()
    with torch.inference_mode(): outputs=model(**inputs,output_hidden_states=True,use_cache=False)
    if DEVICE=='mps': torch.mps.synchronize()
    return outputs.hidden_states[-1][0,-1].float().cpu(),(time.perf_counter()-started)*1000
cal=[]
for row in calibration: v,_=embed(row['text']); cal.append((v,int(row['label'])))
X=torch.stack([v for v,_ in cal]); y=torch.tensor([label for _,label in cal],dtype=torch.float32); mean=X.mean(0); scale=X.std(0).clamp_min(1e-6); X=(X-mean)/scale; X=torch.cat([X,torch.ones(len(X),1)],1); reg=torch.eye(X.shape[1]); reg[-1,-1]=0; weights=torch.linalg.solve(X.T@X+reg,X.T@y)
rows=[]
for row in giskard:
    v,latency=embed(row['prompt']); z=(torch.cat([((v-mean)/scale),torch.ones(1)])@weights).item(); probability=float(torch.sigmoid(torch.tensor(z))); rows.append({'index':row['index'],'name':row['name'],'group':row['group'],'probability_injection':probability,'latency_ms':latency})
thresholds={}
for t in (.5,.6,.7,.8,.9):
    detected=sum(row['probability_injection']>=t for row in rows)
    thresholds[str(t)]={'detected':detected,'recall':detected/len(rows)}
result={'model':MODEL,'dataset':'Giskard-AI/prompt-injections prompt_injections.csv','n':len(rows),'groups':{'Jailbreak':20,'Hijacking attacks':15},'positive_only':True,'thresholds':thresholds,'mean_latency_ms':sum(r['latency_ms'] for r in rows)/len(rows),'p95_latency_ms':sorted(r['latency_ms'] for r in rows)[int(len(rows)*.95)-1]}
Path('giskard_results.json').write_text(json.dumps({'summary':result,'rows':rows},indent=2)); print(json.dumps(result,indent=2))
