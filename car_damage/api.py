"""FastAPI service: POST /analyze (multipart image) -> JSON damage + cost estimate.
Run: uvicorn car_damage.api:app --host 0.0.0.0 --port 8000"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .pipeline import analyze_bytes, get_detector

MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
app = FastAPI(title="Car Damage Estimation API", version="1.1")


@app.get("/", include_in_schema=False)
def home():
    """Serve the project UI, including the Google Maps garage finder."""
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "backend": get_detector().depth_estimator.name}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), overlay: bool = False, include_3d: bool = False):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Upload a JPEG or PNG image.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_BYTES} bytes.")
    try:
        return analyze_bytes(data, include_overlay=overlay, include_3d=include_3d)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
