# METEORA Prototype

High-Resolution Cyclone Downscaling, Physics Verification, and GIS Alert Engine for **Cyclone Biparjoy**.

```
meteora-prototype/
│
├── app.py                # Main Streamlit / Web UI interface
├── api.py                # FastAPI endpoints for JSON alerts
├── data/
│   └── biparjoy_sample.nc# Curated NetCDF slice of Cyclone Biparjoy
├── core/
│   ├── tracker.py        # Stage 1: Centroid & Bounding Box extraction
│   ├── downscaler.py     # Stage 2: 12 km -> 5 km diffusion/enhancement logic
│   └── physics.py        # Mass & moisture flux verification stencil
└── utils/
    └── geojson_gen.py    # Outputs 5 km polygon rings for GIS mapping
```

---

## 🚀 Key Modules & Architecture

1. **Stage 1: Centroid & Bounding Box Tracker (`core/tracker.py`)**
   - Ingests NetCDF pressure fields (`msl`).
   - Extracts sub-grid cyclone vortex centroid and minimum central pressure.
   - Calculates dynamic bounding boxes around the cyclone core.
   - Computes IMD/WMO cyclone categories (e.g., Extremely Severe Cyclonic Storm) and estimated sustained wind speeds.

2. **Stage 2: 12 km ➔ 5 km Diffusion Downscaler (`core/downscaler.py`)**
   - High-order cubic base spatial interpolation to target 5 km grid.
   - Physics-informed reverse diffusion PDE iterations.
   - Recovers eyewall gradient steepening and rainband mesoscale structure while conserving background mean pressure.

3. **Stage 3: Mass & Moisture Flux Stencil (`core/physics.py`)**
   - Geostrophic & gradient cyclonic wind balance stencils.
   - 2D horizontal mass continuity divergence stencil $\nabla \cdot (\rho \mathbf{u})$.
   - Moisture flux convergence (MFC) stencil $-\nabla \cdot (q \rho \mathbf{u})$.
   - Automated Physics Stencil Verification Scorecard (0–100%).

4. **GIS Ring Generator (`utils/geojson_gen.py`)**
   - Generates RFC 7946 compliant GeoJSON features.
   - Produces 5 km, 25 km, 65 km, 150 km, and 250 km risk buffers and bounding boxes for GIS platforms (Leaflet, Mapbox, QGIS, ArcGIS).

5. **JSON Alerts API (`api.py`)**
   - FastAPI server with CORS enabled.
   - Endpoints for emergency civil defense alerts, live tracks, and on-demand 5 km downscaling.

6. **Web UI Interface (`app.py`)**
   - Full interactive Streamlit dashboard with dark-mode visualization, side-by-side coarse vs downscaled comparisons, physics verification scorecards, and live GIS downloads.

---

## 💻 Quick Start

### 1. Run the Streamlit Dashboard
```bash
cd meteora-prototype
streamlit run app.py
```

### 2. Run the FastAPI Alert Server
```bash
cd meteora-prototype
uvicorn api:app --reload --port 8000
```
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Live Cyclone JSON Alert: `http://localhost:8000/api/v1/alerts`
- GIS GeoJSON 5 km Rings: `http://localhost:8000/api/v1/geojson`
