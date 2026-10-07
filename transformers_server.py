import base64, io, os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.getenv('CLEF_MODEL', '/Users/Shabul/model-weights/Cloudflare/clef-flash' if Path('/Users/Shabul/model-weights/Cloudflare/clef-flash').exists() else 'Cloudflare/clef-flash')
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
app = FastAPI()
ROOT = Path(__file__).parent
app.mount('/static', StaticFiles(directory=ROOT), name='static')
print(f'Loading {MODEL_ID} on {DEVICE} …', flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
).to(DEVICE)
model.eval()
print('Model ready on http://127.0.0.1:3000', flush=True)

@app.get('/')
async def index(): return FileResponse(ROOT / 'index.html')
@app.get('/styles.css')
async def styles(): return FileResponse(ROOT / 'styles.css', media_type='text/css')
@app.get('/app.js')
async def script(): return FileResponse(ROOT / 'app.js', media_type='application/javascript')

@app.get('/health')
async def health(): return {'ok': True, 'model': MODEL_ID, 'device': DEVICE}

@app.post('/api/chat')
async def chat(request: Request):
    payload = await request.json()
    messages = payload.get('messages', [])
    normalized = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get('type') == 'text': parts.append({'type': 'text', 'text': part['text']})
                elif part.get('type') == 'image_url': parts.append({'type': 'image', 'url': part['image_url']['url']})
            content = parts
        normalized.append({'role': msg['role'], 'content': content})
    inputs = processor.apply_chat_template(normalized, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
    inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.inference_mode():
        temperature = float(payload.get('temperature', .7))
        out = model.generate(**inputs, max_new_tokens=int(payload.get('max_tokens', 512)), temperature=max(temperature, 0.01), do_sample=temperature > 0)
    text = processor.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return {'id': 'clef-local', 'object': 'chat.completion', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=3000)
