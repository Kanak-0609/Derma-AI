"""FastAPI service. Run from the repo root: uvicorn api.app:app --port 8000   (docs at /docs)"""
import base64, io, os, sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from serving import DEFAULT_BUNDLE, Service, render

state = {}
MAX_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    state['svc'] = Service(os.environ.get('DERMAAI_BUNDLE', str(DEFAULT_BUNDLE)))
    yield
    state.clear()


app = FastAPI(title='DermaAI decision-support API (research prototype, not a medical device)',
              version='1.0', lifespan=lifespan)


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


@app.get('/health')
def health():
    return {'status': 'ok', 'model': state['svc'].info}


@app.post('/predict')
async def predict(file: UploadFile = File(...), explain: bool = True):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image is larger than 10 MB')
    try:
        pil = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        raise HTTPException(400, 'File is not a readable image')
    r = state['svc'].predict(pil, explain=explain)      # async endpoint: requests run one at a time (Grad-CAM hooks are not thread-safe)
    out = {k: r[k] for k in ('predicted_class', 'confidence', 'probabilities', 'referred_for_review', 'message')}
    if explain:
        for name, img in render(pil, r).items():
            out[f'{name}_png_base64'] = _b64(img)
    return out
