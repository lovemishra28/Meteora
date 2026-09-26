"""
FastAPI endpoints for Cyclone Biparjoy JSON alerts, GIS GeoJSON streams,
and AI Downscaling / Physics verification stencils.
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.tracker import CycloneTracker
from core.downscaler import DiffusionDownscaler
from core.physics import PhysicsVerifier
from utils.geojson_gen import generate_cyclone_geojson

# Locate sample dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NC_PATH = os.path.join(DATA_DIR, "biparjoy_sample.nc")

app = FastAPI(
    title="Meteora Prototype API",
    description="Cyclone Biparjoy Downscaling, Physics Stencil Verification & Real-time GIS Alerts",
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

# Global core service singletons
tracker = CycloneTracker(NC_PATH)
downscaler = DiffusionDownscaler(coarse_res_km=12.0, target_res_km=5.0)
verifier = PhysicsVerifier()


class DownscaleRequest(BaseModel):
    time_index: int = Field(default=0, ge=0, description="Time index within dataset")
    diffusion_steps: int = Field(default=10, ge=1, le=50, description="Diffusion steps")
    apply_diffusion: bool = Field(default=True, description="Enable diffusion sharpening")


@app.get("/")
def get_root() -> Dict[str, Any]:
    """Service status and API directory."""
    return {
        "service": "Meteora Cyclone Intelligence Engine",
        "dataset": "Cyclone Biparjoy (Arabian Sea)",
        "total_timesteps": len(tracker.times),
        "status": "operational",
        "endpoints": {
            "track": "/api/v1/track",
            "latest_alert": "/api/v1/alerts",
            "status_by_time": "/api/v1/status?time_index=0",
            "geojson_rings": "/api/v1/geojson?time_index=0",
            "downscale_and_verify": "POST /api/v1/downscale",
            "docs": "/docs"
        }
    }


@app.get("/api/v1/track")
def get_cyclone_track() -> Dict[str, Any]:
    """Retrieve full temporal trajectory and intensity history."""
    all_tracks = tracker.track_all()
    return {
        "storm_name": "Cyclone Biparjoy",
        "basin": "North Indian Ocean (Arabian Sea)",
        "count": len(all_tracks),
        "track": all_tracks
    }


@app.get("/api/v1/status")
def get_timestep_status(
    time_index: int = Query(default=0, ge=0, description="Timestep index")
) -> Dict[str, Any]:
    """Get centroid, bounding box, and intensity metrics for a specific timestep."""
    if time_index >= len(tracker.times):
        raise HTTPException(
            status_code=400,
            detail=f"time_index {time_index} exceeds max index {len(tracker.times) - 1}"
        )
    return tracker.process_timestep(time_index)


@app.get("/api/v1/alerts")
def get_json_alerts(
    time_index: Optional[int] = Query(default=None, description="Timestep index (defaults to latest)")
) -> Dict[str, Any]:
    """
    Generate structured emergency JSON alert for civil defense and coastal authorities.
    """
    idx = time_index if time_index is not None else len(tracker.times) - 1
    if idx < 0 or idx >= len(tracker.times):
        raise HTTPException(status_code=400, detail="Invalid time_index")

    step_info = tracker.process_timestep(idx)
    centroid = step_info["centroid"]
    min_p = step_info["min_pressure_hpa"]
    intensity = step_info["intensity"]

    # Generate warnings based on alert level
    alerts_bulletin = []
    if intensity["alert_level"] in ["CODE_RED", "CODE_RED_EXTREME"]:
        alerts_bulletin.append("EXTREME DANGER: Evacuation required within 65 km destructive radius.")
        alerts_bulletin.append("Storm surge warning: Coastal inundation potential 2.5 - 4.5m.")
        alerts_bulletin.append("Fishermen advised not to venture into Central and North Arabian Sea.")
    elif intensity["alert_level"] == "CODE_ORANGE":
        alerts_bulletin.append("SEVERE ALERT: Torrential rainfall and gale wind speeds up to 100 km/h.")
        alerts_bulletin.append("Secure weak structures, hoardings, and suspended utility lines.")
    else:
        alerts_bulletin.append("ADVISORY: Tropical disturbance monitored by Meteora high-res radar pipeline.")

    return {
        "alert_id": f"METEORA-ALERT-BIPARJOY-{idx:03d}",
        "timestamp": step_info["timestamp"],
        "storm": "Cyclone Biparjoy",
        "severity": intensity["alert_level"],
        "category": intensity["category"],
        "central_pressure_hpa": min_p,
        "max_sustained_wind_kmh": intensity["vmax_kmh"],
        "max_sustained_wind_knots": intensity["vmax_knots"],
        "eye_coordinates": centroid,
        "bounding_box_wgs84": step_info["bounding_box"],
        "emergency_bulletin": alerts_bulletin,
        "recommended_actions": [
            "Trigger automated 5 km ring evacuation buffers in regional GIS",
            "Deploy emergency medical teams outside the 65 km destructive zone",
            "Monitor 5 km downscaled rainband updates every 15 minutes"
        ]
    }


@app.get("/api/v1/geojson")
def get_geojson_layers(
    time_index: int = Query(default=0, ge=0, description="Timestep index")
) -> Dict[str, Any]:
    """
    Outputs 5 km polygon rings and storm center for Leaflet, Mapbox, and GIS mapping.
    """
    if time_index >= len(tracker.times):
        raise HTTPException(status_code=400, detail="time_index out of bounds")

    step = tracker.process_timestep(time_index)
    return generate_cyclone_geojson(
        centroid_lat=step["centroid"]["lat"],
        centroid_lon=step["centroid"]["lon"],
        min_pressure_hpa=step["min_pressure_hpa"],
        timestamp=step["timestamp"],
        intensity_info=step["intensity"],
        bbox=step["bounding_box"]
    )


@app.post("/api/v1/downscale")
def run_downscale_and_verify(req: DownscaleRequest) -> Dict[str, Any]:
    """
    Execute Stage 2 (12 km -> 5 km Diffusion Downscaling) and
    Stage 3 (Mass & Moisture Flux Physics Verification Stencil).
    """
    if req.time_index >= len(tracker.times):
        raise HTTPException(status_code=400, detail="Invalid time_index")

    pressure_2d, lats, lons, ts_str = tracker.get_slice(req.time_index)

    # 1. Downscale
    ds_engine = DiffusionDownscaler(
        coarse_res_km=12.0,
        target_res_km=5.0,
        diffusion_steps=req.diffusion_steps
    )
    downscale_res = ds_engine.downscale(
        coarse_field=pressure_2d,
        coarse_lats=lats,
        coarse_lons=lons,
        apply_diffusion=req.apply_diffusion
    )

    # 2. Physics verification on downscaled field
    hr_p = downscale_res["hr_field"]
    hr_lats = downscale_res["hr_lats"]
    hr_lons = downscale_res["hr_lons"]
    phys_res = verifier.verify_field(hr_p, hr_lats, hr_lons)

    # Clean out large arrays from JSON response (summary stats only)
    return {
        "timestamp": ts_str,
        "coarse_resolution": f"{downscale_res['coarse_res_km']} km",
        "target_resolution": f"{downscale_res['target_res_km']} km",
        "coarse_shape": list(downscale_res["coarse_shape"]),
        "downscaled_shape": list(downscale_res["hr_shape"]),
        "diffusion_steps_applied": downscale_res["diffusion_steps_applied"],
        "gradient_enhancement_ratio": downscale_res["gradient_enhancement_ratio"],
        "max_gradient_coarse": downscale_res["max_gradient_coarse"],
        "max_gradient_enhanced": downscale_res["max_gradient_hr"],
        "physics_verification": {
            "physics_consistency_score": phys_res["physics_consistency_score"],
            "is_physically_consistent": phys_res["is_physically_consistent"],
            "max_wind_knots": phys_res["max_wind_knots"],
            "mean_mass_divergence_residual": phys_res["mean_mass_divergence_residual"],
            "core_moisture_flux_convergence": phys_res["core_moisture_flux_convergence"],
            "max_moisture_flux": phys_res["max_moisture_flux"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
