# api.py
import os
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from core.tracker import AnomalyTracker
from utils.geojson_gen import generate_cyclone_geojson
import uvicorn

# Live ingestion state
_ingest_lock   = threading.Lock()
_ingest_running = False

# FastAPI App Initialization
app = FastAPI(
    title="Meteora Alerting API",
    description="REST API to serve pinpoint 5km weather anomalies & threat footprints to NDRF.",
    version="1.0.0"
)

# Enable CORS for web dashboards and GIS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tracker engine globally load kar rahe hain taaki har request fast ho
try:
    tracker = AnomalyTracker()
except Exception as e:
    tracker = None

@app.get("/")
def read_root():
    return {"status": "Meteora API is running. Core systems online."}

@app.get("/api/v1/alert/{time_idx}")
def get_disaster_alert(time_idx: int):
    """
    NDRF aur SDMA ke liye automated JSON alert generate karta hai.
    time_idx: 0 se 6 tak (representing 12-hour forecast lead times)
    """
    if tracker is None:
        raise HTTPException(status_code=500, detail="Data engine offline. Pehle data_loader.py run karein.")
    
    if time_idx < 0 or time_idx >= len(tracker.times):
        raise HTTPException(
            status_code=404,
            detail=f"Time index range se bahar hai. Valid range: 0-{len(tracker.times) - 1}."
        )

    try:
        # Core tracker se storm ka exact location aur details nikalna
        storm_data = tracker.detect_storm_center(time_idx)
        
        # NDRF ke liye CAP-compliant (Common Alerting Protocol) style alert payload
        alert_payload = {
            "alert_id": f"BIPARJOY-T{time_idx}",
            "timestamp_utc": storm_data["timestamp"],
            "severity": storm_data["alert_level"],
            "system_category": storm_data["category"],
            "impact_zone": {
                "threat_type": "Pinpoint 5km localized impact",
                "centroid_lat": storm_data["centroid"][0],
                "centroid_lon": storm_data["centroid"][1],
                "bounding_box": storm_data["bbox"]
            },
            "peak_metrics": {
                "min_pressure_hpa": storm_data["min_mslp_hpa"],
                "max_wind_kmh": storm_data["max_wind_kmh"],
                "max_rainfall_mmh": storm_data["max_precip_mmh"]
            },
            "ndrf_action_protocol": "Deploy rescue boats and de-watering pumps to exact centroid coordinates." if storm_data["alert_level"] in ["CRITICAL", "HIGH"] else "Monitor trajectory."
        }
        return alert_payload
        
    except IndexError:
        raise HTTPException(status_code=404, detail="Time index range se bahar hai. Valid range: 0-6.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/track")
def get_full_trajectory():
    """
    Returns complete multi-timestep trajectory for GIS mapping and forecasting.
    """
    if tracker is None:
        raise HTTPException(status_code=500, detail="Data engine offline.")
    return {
        "storm": "Cyclone Biparjoy",
        "basin": "Arabian Sea",
        "timesteps": len(tracker.times),
        "trajectory": tracker.get_full_trajectory()
    }

@app.get("/api/v1/geojson/{time_idx}")
def get_geojson_layers(time_idx: int = 0):
    """
    Returns 5 km GIS GeoJSON concentric risk rings and center point for Leaflet/Mapbox.
    """
    if tracker is None:
        raise HTTPException(status_code=500, detail="Data engine offline.")
    if time_idx < 0 or time_idx >= len(tracker.times):
        raise HTTPException(status_code=404, detail="Time index range se bahar hai.")

    storm_data = tracker.detect_storm_center(time_idx)
    bbox_list = [
        storm_data["bbox"]["lon_min"],
        storm_data["bbox"]["lat_min"],
        storm_data["bbox"]["lon_max"],
        storm_data["bbox"]["lat_max"],
    ]
    intensity_info = {
        "category": storm_data["category"],
        "alert_level": storm_data["alert_level"],
        "vmax_kmh": storm_data["max_wind_kmh"],
        "vmax_knots": round(storm_data["max_wind_kmh"] / 1.852, 1)
    }
    return generate_cyclone_geojson(
        centroid_lat=storm_data["centroid"][0],
        centroid_lon=storm_data["centroid"][1],
        min_pressure_hpa=storm_data["min_mslp_hpa"],
        timestamp=storm_data["timestamp"],
        intensity_info=intensity_info,
        bbox=bbox_list
    )


# ─── Live Ingestion Endpoints ─────────────────────────────────────────────────

@app.post("/api/v1/ingest")
async def trigger_live_ingest(background_tasks: BackgroundTasks, force: bool = False):
    """
    Trigger a live weather data ingestion run in the background.

    Sources tried (in order):
      1. Open-Meteo (free JSON API, no auth)
      2. GFS NOMADS OpenDAP (NCEP operational NWP)
      3. ERA5 CDS (requires ~/.cdsapirc credentials)
      4. Static biparjoy_real.nc (always-available fallback)

    Results cached to data/live_latest.nc for 3 hours.
    Subsequent calls within the TTL return cached status immediately.
    """
    global _ingest_running
    with _ingest_lock:
        if _ingest_running:
            return {"status": "running", "message": "Ingestion already in progress."}
        _ingest_running = True

    def _run():
        global _ingest_running
        try:
            from core.live_ingestion import LiveIngestionPipeline
            pipeline = LiveIngestionPipeline()
            pipeline.fetch(force=force)
        except Exception as exc:
            print(f"[api/ingest] background task error: {exc}")
        finally:
            with _ingest_lock:
                _ingest_running = False

    background_tasks.add_task(_run)
    return {
        "status":  "started",
        "message": "Live ingestion started in background. Poll /api/v1/ingest/status for result.",
        "force":   force,
    }


@app.get("/api/v1/ingest/status")
def get_ingest_status():
    """
    Returns the status of the most recent live ingestion run.
    """
    global _ingest_running
    try:
        from core.live_ingestion import LiveIngestionPipeline
        status = LiveIngestionPipeline().get_status()
    except Exception as exc:
        status = {"error": str(exc)}
    status["currently_running"] = _ingest_running
    return status


@app.get("/api/v1/model/status")
def get_model_status():
    """
    Returns the status of the Physics-Guided Diffusion Model (PGDM).
    Shows whether the real DDIM model is active or scipy mock is running.
    """
    weights_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weights", "meteora_diff_biparjoy.pt"
    )
    weights_exist = os.path.exists(weights_path)
    weights_size_mb = os.path.getsize(weights_path) / 1e6 if weights_exist else 0

    try:
        from core.downscaler import DownscalerEngine
        mode = DownscalerEngine().mode
    except Exception:
        mode = "UNKNOWN"

    return {
        "mode":            mode,
        "weights_exist":   weights_exist,
        "weights_size_mb": round(weights_size_mb, 1),
        "weights_path":    weights_path,
        "description": (
            "Physics-Guided Diffusion Model (PGDM) active — real DDIM inference"
            if mode == "PGDM_REAL" else
            "scipy mock active — run scripts/train_diffusion.py to activate PGDM"
        ),
        "train_command":  "python scripts/train_diffusion.py --epochs 50",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

