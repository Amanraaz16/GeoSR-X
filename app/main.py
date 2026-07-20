"""
GeoSR-X Platform — FastAPI Backend
Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
import os

# Make project root importable (model.py, losses.py etc. are at root level)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes.process  import router as process_router
from app.routes.download import router as download_router
from app.core.inference  import load_models

app = FastAPI(title="GeoSR-X", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models at startup
@app.on_event("startup")
async def startup():
    load_models()

# API routes
app.include_router(process_router,  prefix="/api")
app.include_router(download_router, prefix="/api")

# Serve React frontend
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}