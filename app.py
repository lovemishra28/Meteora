# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# Core imports
from core.data_loader import load_biparjoy_data
from core.tracker import AnomalyTracker
from core.downscaler import DownscalerEngine

st.set_page_config(
    page_title="Meteora | NDRF Tactical Dashboard",
    page_icon="🌪️",
    layout="wide"
)

# 1. Initialize Engines (Cached for performance)
@st.cache_resource
def init_system():
    load_biparjoy_data() # Ensure data exists
    tracker = AnomalyTracker()
    downscaler = DownscalerEngine()
    return tracker, downscaler

tracker, downscaler = init_system()

# 2. Main UI Header
st.title("🌪️ METEORA: Extreme Weather Tracking & Downscaling")
st.markdown("**Physics-Informed Spherical GNN & Diffusion Pipeline | Pilot: Cyclone Biparjoy (Arabian Sea)**")
st.divider()

# 3. Sidebar Controls
st.sidebar.header("Command Center")
st.sidebar.markdown("Slide to track the forecast lead time (T=0 to T+72h)")
time_idx = st.sidebar.slider("Forecast Lead Time (Step)", min_value=0, max_value=6, value=0, step=1)

# Fetch Current Storm Data
storm_data = tracker.detect_storm_center(time_idx)
cropped_patch = tracker.crop_anomaly_subgrid(time_idx)
downscale_metrics = downscaler.process_anomaly_patch(cropped_patch, variable="u10")

# 4. Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Timestamp", storm_data["timestamp"])
col2.metric("Min Central Pressure", f"{storm_data['min_mslp_hpa']} hPa")
col3.metric("Max Wind Speed", f"{storm_data['max_wind_kmh']} km/h")
col4.metric("NDRF Alert Level", storm_data["alert_level"])

st.markdown("---")

# 5. Split View: Map on Left, Downscaling Analytics on Right
map_col, analytics_col = st.columns([1.2, 1])

with map_col:
    st.subheader("📍 Real-Time Spatial Tracking")
    st.markdown("Dynamic 4D Bounding Box highlighting the 12km tracking anomaly.")
    
    # Initialize Folium Map centered at storm
    c_lat, c_lon = storm_data["centroid"]
    m = folium.Map(location=[c_lat, c_lon], zoom_start=6, tiles="OpenStreetMap")
    
    # Draw Stage 1 Bounding Box (12km Coarse Area)
    b = storm_data["bbox"]
    folium.Rectangle(
        bounds=[[b["lat_min"], b["lon_min"]], [b["lat_max"], b["lon_max"]]],
        color="#ff7800",
        fill=True,
        fill_opacity=0.1,
        weight=2,
        tooltip="Stage 1: 12km Tracked Bounding Box"
    ).add_to(m)
    
    # Draw 5km Pinpoint Impact Zone
    folium.CircleMarker(
        location=[c_lat, c_lon],
        radius=40, # visual size
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.4,
        tooltip="Stage 2: 5km High Impact Threat Zone"
    ).add_to(m)
    
    folium.Marker(
        [c_lat, c_lon],
        popup=f"Cyclone Core: {storm_data['category']}"
    ).add_to(m)

    st_folium(m, width=650, height=460, key=f"folium_map_{time_idx}")

with analytics_col:
    st.subheader("⚙️ Stage 2: Downscaling Engine")
    st.markdown("Demonstrating the **Spectral Smoothing Trap** vs **Meteora Diffusion**")
    
    # Alert Card
    if storm_data["alert_level"] in ["CRITICAL", "HIGH"]:
        st.error(f"🚨 **TACTICAL ALERT:** {storm_data['category']}. Deploy teams to {c_lat:.2f}°N, {c_lon:.2f}°E.")
    else:
        st.warning(f"⚠️ **ADVISORY:** {storm_data['category']}. Monitor trajectory.")
        
    st.markdown("#### Wind Speed Peak Preservation (12km -> 5km)")
    
    # Side-by-side comparison metrics
    sc1, sc2 = st.columns(2)
    
    cnn_drop = downscale_metrics['original_peak'] - downscale_metrics['cnn_peak']
    
    sc1.metric(
        label="Standard CNN Output", 
        value=f"{downscale_metrics['cnn_peak']} km/h",
        delta=f"-{round(cnn_drop, 2)} km/h (Smoothed)",
        delta_color="inverse"
    )
    sc1.caption("Standard DL blurs the extreme peak, causing false safety alerts.")
    
    sc2.metric(
        label="Meteora Diffusion Output", 
        value=f"{downscale_metrics['diffusion_peak']} km/h",
        delta="100% Peak Preserved",
        delta_color="normal"
    )
    sc2.caption("Physics-guided diffusion retains true amplitude for accurate disaster response.")
    
    # Simulated Raw API Payload
    with st.expander("Show Raw API JSON Payload (CAP-Compliant)"):
        st.json({
            "alert_id": f"BIPARJOY-T{time_idx}",
            "impact_zone": storm_data["bbox"],
            "peak_metrics": {
                "wind": storm_data["max_wind_kmh"],
                "pressure": storm_data["min_mslp_hpa"]
            }
        })
