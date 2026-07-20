"""
POST /api/process  — accepts a GeoTIFF, runs all 4 SR models, returns metrics.
POST /api/process/latlon — accepts lat/lon/date, searches Copernicus catalogue.
"""
import os
import uuid
import shutil
import json
import asyncio
import math
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.inference import process_geotiff

def sanitise_for_json(obj):
    """
    Recursively replace float NaN and Inf with None so the
    response is always valid JSON. NaN appears when metrics
    are computed on tiles with large nodata regions.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_for_json(v) for v in obj]
    return obj

router = APIRouter()

JOBS_DIR = "app/jobs"
os.makedirs(JOBS_DIR, exist_ok=True)


@router.post("/process")
async def process_upload(file: UploadFile = File(...)):
    """
    Accept a GeoTIFF upload and run all 4 SR models.
    Returns a Server-Sent Events stream with progress updates,
    then a final JSON result as the last event.
    """
    if not file.filename.endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files (.tif/.tiff) are accepted.")

    job_id   = str(uuid.uuid4())[:8]
    job_dir  = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, file.filename)
    content    = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # Collect progress messages in a list so the generator can stream them
    progress_messages = []
    result_holder     = {}
    error_holder      = {}

    def on_progress(msg: str):
        progress_messages.append(msg)

    # Run processing in a thread so SSE stream stays alive
    import threading
    def run():
        try:
            metrics, output_paths, metadata, ndvi_b64, ndvi_tif_path = \
                process_geotiff(input_path, job_dir,
                                progress_callback=on_progress)
            result_holder["data"] = {
                "job_id":       job_id,
                "metadata":     metadata,
                "metrics":      metrics,
                "models":       [m for m in output_paths.keys()
                                  if m != "ndvi_diff"],
                "ndvi_b64":     ndvi_b64,
                "has_ndvi_tif": ndvi_tif_path is not None,
            }
        except ValueError as e:
            # Known errors (band validation, etc) — show directly to user
            msg = str(e)
            error_holder["msg"] = msg
            on_progress(f"Stopped: {msg}")
        except Exception as e:
            # Unknown errors — log full details server-side, show clean msg
            import traceback
            tb = traceback.format_exc()
            print(f"[GeoSR-X] Processing error:\n{tb}")
            error_holder["msg"] = (
                f"{str(e)} — Check server logs for details. "
                f"Common causes: unsupported band order, corrupt GeoTIFF, "
                f"or insufficient memory for large tiles."
            )
            on_progress(f"Stopped: {str(e)}")

    thread = threading.Thread(target=run)
    thread.start()

    async def event_stream() -> AsyncGenerator[str, None]:
        sent = 0
        try:
            while thread.is_alive() or sent < len(progress_messages):
                while sent < len(progress_messages):
                    msg = progress_messages[sent]
                    yield f"data: {json.dumps({'step': msg})}\n\n"
                    sent += 1
                await asyncio.sleep(0.2)

            # Drain any final messages after thread finishes
            while sent < len(progress_messages):
                msg = progress_messages[sent]
                yield f"data: {json.dumps({'step': msg})}\n\n"
                sent += 1

            if error_holder:
                shutil.rmtree(job_dir, ignore_errors=True)
                yield f"data: {json.dumps({'error': error_holder['msg']})}\n\n"
            elif "data" in result_holder:
                clean = sanitise_for_json(result_holder["data"])
                yield f"data: {json.dumps({'result': clean})}\n\n"
            else:
                # Thread finished but put nothing in either holder
                # This means an unhandled exception in the thread
                shutil.rmtree(job_dir, ignore_errors=True)
                yield f"data: {json.dumps({'error': 'Processing failed with an unexpected error. Check that your GeoTIFF contains bands B2, B3, B4, B8 at 10m resolution.'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'Stream error: {str(e)}'})}\n\n"

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")


@router.post("/process/latlon")
async def process_latlon(
    lat:  float = Form(...),
    lon:  float = Form(...),
    date: str   = Form(...),
):
    """
    Search Copernicus Data Space for Sentinel-2 scenes near a lat/lon/date.
    Returns scene info + browser link. Does not auto-download
    (requires user account for actual download).
    """
    import requests
    from datetime import datetime, timedelta

    # Parse date and build a +-3 day window to maximise chance of finding a scene
    try:
        centre = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Date must be in YYYY-MM-DD format.")

    start = (centre - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00.000Z")
    end   = (centre + timedelta(days=3)).strftime("%Y-%m-%dT23:59:59.999Z")

    # Fix 2: clean OData query — no cloud filter in URL (caused syntax errors)
    search_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq 'SENTINEL-2'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;"
        f"POINT({lon} {lat})')"
        f" and ContentDate/Start gt {start}"
        f" and ContentDate/Start lt {end}"
        f"&$orderby=ContentDate/Start desc&$top=10"
    )

    try:
        resp = requests.get(search_url, timeout=20,
                            headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise HTTPException(503,
            "Copernicus catalogue timed out. Try again in a moment.")
    except Exception as e:
        raise HTTPException(503,
            f"Copernicus catalogue query failed: {e}. "
            f"Upload a GeoTIFF directly instead.")

    products = data.get("value", [])

    if not products:
        raise HTTPException(404,
            f"No Sentinel-2 scenes found within 3 days of {date} "
            f"at lat={lat}, lon={lon}. "
            f"Try a different date — Sentinel-2 revisits every ~5 days.")

    # Pick least cloudy from results
    def cloud_cover(p):
        for a in p.get("Attributes", []):
            if a.get("Name") == "cloudCover":
                return float(a.get("Value", 100))
        return 100.0

    products_sorted = sorted(products, key=cloud_cover)
    best = products_sorted[0]
    cc   = cloud_cover(best)

    # Build direct download URL for the best product
    product_id = best.get("Id", "")
    direct_download_url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    product_id   = best.get("Id", "")
    product_name = best.get("Name", "")
    scene_date   = best.get("ContentDate", {}).get("Start", "")[:10]

    browser_deep_link = (
        f"https://browser.dataspace.copernicus.eu/"
        f"?zoom=12&lat={lat}&lng={lon}"
        f"&dateMode=SINGLE&date={scene_date}"
        f"&cloudCoverage=100"
        f"&datasetId=S2_L2A_CDAS"
    )

    # Extract MGRS tile ID from product name for GEE filtering
    # Product name format: S2B_MSIL2A_20250930T052649_N0511_R105_T43REK_...
    mgrs_tile = ""
    parts = product_name.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            mgrs_tile = p[1:]  # strip leading T -> "43REK"
            break

    # Generate ready-to-paste GEE script
    gee_script = f"""// ============================================================
// GeoSR-X: Auto-generated GEE export script
// Scene: {product_name}
// Date:  {scene_date}
// Tile:  {mgrs_tile}
// ============================================================

// 5×5 km square centred on the target point (geodesic:false = planar, exact size)
var centre = ee.Geometry.Point([{lon}, {lat}]);
var roi    = centre.buffer({{distance: 2500, proj: 'EPSG:32643'}}).bounds(1, 'EPSG:32643');

// Search BOTH L1C and L2A collections with a wider date window
// The Copernicus catalogue found: {product_name}
// Using 7-day window to maximise chance of finding the scene in GEE
var startDate = ee.Date('{scene_date}').advance(-3, 'day');
var endDate   = ee.Date('{scene_date}').advance(4, 'day');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));

print('L2A scenes found in window:', s2.size());

// If L2A returns 0, try L1C (older scenes may only have L1C in GEE)
var s2_l1c = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));

print('L1C scenes found in window:', s2_l1c.size());

// Use whichever collection has data (prefer L2A)
var collection = ee.Algorithms.If(
  s2.size().gt(0), s2, s2_l1c
);
var col = ee.ImageCollection(collection);

var image = col.sort('CLOUDY_PIXEL_PERCENTAGE').first();
print('Selected scene:', image.get('system:index'));
print('Cloud %:', image.get('CLOUDY_PIXEL_PERCENTAGE'));

// Cloud mask using SCL (L2A only — skipped for L1C automatically)
function maskClouds(img) {{
  var bands = img.bandNames();
  var hasSCL = bands.contains('SCL');
  var scl  = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return ee.Algorithms.If(hasSCL, img.updateMask(mask), img);
}}

var masked = ee.Image(maskClouds(image)).select(['B2','B3','B4','B8']);

Map.centerObject(roi, 13);
var clipped = masked.clip(roi);
Map.addLayer(
  clipped,
  {{bands: ['B4','B3','B2'], min: 0, max: 3000}},
  'True Color'
);

// Clip image to exact ROI before export
var clipped = masked.clip(roi);

Export.image.toDrive({{
  image: clipped,
  description: 'GeoSRX_{mgrs_tile}_{scene_date.replace("-","")}',
  folder: 'GeoSRX_data',
  fileNamePrefix: 'S2_MSIL2A_{mgrs_tile}_{scene_date.replace("-","")}_4band',
  region: roi.getInfo(),
  scale: 10,
  crs: 'EPSG:32643',
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {{cloudOptimized: true}}
}});

print('Check console for scene count. If both show 0, try a different date.');
print('Go to Tasks tab and click RUN to export.');
"""

    return JSONResponse({
        "message": (
            "Scene found. Use Copernicus Browser or the GEE script below "
            "to download the tile."
        ),
        "product_name": product_name,
        "product_id":   product_id,
        "date":         scene_date,
        "cloud_cover":  round(cc, 1),
        "scenes_found": len(products),
        "browser_url":  browser_deep_link,
        "gee_script":   gee_script,
        "mgrs_tile":    mgrs_tile,
    })
@router.post("/preview")
async def preview_upload(file: UploadFile = File(...)):
    """
    Generate a quick RGB thumbnail from an uploaded GeoTIFF for
    visual confirmation before full processing. Returns base64 PNG.
    Does not save the file permanently — reads it in memory only.
    """
    import base64
    import io
    import numpy as np
    import rasterio
    from PIL import Image

    if not file.filename.endswith((".tif", ".tiff")):
        raise HTTPException(400, "Only GeoTIFF files accepted.")

    contents = await file.read()

    try:
        with rasterio.open(io.BytesIO(contents)) as src:
            arr = src.read().astype(np.float32)
            band_count = src.count
    except Exception as e:
        raise HTTPException(422, f"Could not read GeoTIFF: {e}")

    # Use first 3 bands as RGB (B2,B3,B4 in our case)
    if band_count >= 3:
        rgb = arr[:3]
    else:
        rgb = np.stack([arr[0]] * 3, axis=0)

    # Percentile stretch for display
    p2  = np.percentile(rgb[rgb > 0], 2)  if (rgb > 0).any() else 0
    p98 = np.percentile(rgb[rgb > 0], 98) if (rgb > 0).any() else 1
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    else:
        rgb = np.clip(rgb, 0, 1)

    # Resize to thumbnail
    h, w = rgb.shape[1], rgb.shape[2]
    max_dim = 512
    scale = min(max_dim / h, max_dim / w, 1.0)
    th, tw = max(1, int(h * scale)), max(1, int(w * scale))

    img_arr = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="RGB").resize((tw, th), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "filename":    file.filename,
        "band_count":  band_count,
        "height":      h,
        "width":       w,
        "preview_b64": b64,
    })