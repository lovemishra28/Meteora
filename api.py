# api.py
import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.tracker import AnomalyTracker
from utils.geojson_gen import generate_cyclone_geojson
import uvicorn

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
        raise HTTPException(status_code=404, detail="Time index range se bahar hai. Valid range: 0-6.")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
