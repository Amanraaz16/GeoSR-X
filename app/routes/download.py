"""
GET /api/download/{job_id}/{model_name} — download a processed GeoTIFF.
"""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
JOBS_DIR = "app/jobs"


@router.get("/download/{job_id}/{model_name}")
async def download_result(job_id: str, model_name: str):
    valid_models = {"bicubic", "lanczos", "cnn_mse", "geosrx", "ndvi_diff"}
    if model_name not in valid_models:
        raise HTTPException(400, f"Unknown model. Choose from: {valid_models}")

    if model_name == "ndvi_diff":
        path = os.path.join(JOBS_DIR, job_id, "ndvi_diff.tif")
    else:
        path = os.path.join(JOBS_DIR, job_id, f"{model_name}_sr.tif")

    if not os.path.exists(path):
        raise HTTPException(404, f"Result file not found: {path}")

    return FileResponse(
        path,
        media_type="image/tiff",
        filename=f"geosrx_{model_name}_{job_id}.tif",
    )