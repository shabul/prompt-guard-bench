import os
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
ROOT = Path(__file__).parent
app.mount('/static', StaticFiles(directory=ROOT), name='static')

@app.get('/')
async def index(): return FileResponse(ROOT / 'index.html')

@app.get('/health')
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get('http://127.0.0.1:8000/v1/models')
        return {'ok': r.is_success}
    except Exception:
        return JSONResponse({'ok': False}, status_code=503)

@app.post('/api/chat')
async def chat(request: Request):
    payload = await request.json()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post('http://127.0.0.1:8000/v1/chat/completions', json=payload)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({'detail': str(e)}, status_code=502)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=3000)
