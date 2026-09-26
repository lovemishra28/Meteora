"""
METEORA Prototype — Streamlit Web UI Interface
Interactive dashboard for Cyclone Biparjoy tracking, 12 km -> 5 km diffusion downscaling,
physics flux verification stencils, and GIS GeoJSON ring alerts.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.tracker import CycloneTracker
from core.downscaler import DiffusionDownscaler
from core.physics import PhysicsVerifier
from utils.geojson_gen import generate_cyclone_geojson, export_geojson_string

# Page Configuration
st.set_page_config(
    page_title="Meteora | Cyclone Biparjoy Intelligence",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .alert-badge-red {
        background-color: rgba(244, 67, 54, 0.2);
        color: #ff5252;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #f44336;
        display: inline-block;
    }
    .alert-badge-orange {
        background-color: rgba(255, 152, 0, 0.2);
        color: #ffa726;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ff9800;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Dataset path resolution
DEFAULT_NC_PATH = os.path.join(os.path.dirname(__file__), "data", "biparjoy_sample.nc")

@st.cache_resource
def load_models(nc_file: str):
    tracker = CycloneTracker(nc_file)
    downscaler = DiffusionDownscaler(coarse_res_km=12.0, target_res_km=5.0)
    verifier = PhysicsVerifier()
    return tracker, downscaler, verifier

if not os.path.exists(DEFAULT_NC_PATH):
    st.error(f"Dataset not found at `{DEFAULT_NC_PATH}`. Please check file structure.")
    st.stop()

tracker, downscaler, verifier = load_models(DEFAULT_NC_PATH)
num_steps = len(tracker.times)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/cyclone.png", width=64)
st.sidebar.title("Meteora Controls")
st.sidebar.markdown("**Basin**: Arabian Sea | **Target**: Cyclone Biparjoy")

time_idx = st.sidebar.slider(
    "Select Timestep",
    min_value=0,
    max_value=num_steps - 1,
    value=min(12, num_steps - 1),
    help="Navigate through available 6-hourly NetCDF observational intervals."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Stage 2 Downscaler Settings")
diff_steps = st.sidebar.slider("Diffusion Steps", min_value=1, max_value=25, value=12)
apply_diff = st.sidebar.toggle("Physics-informed Diffusion Sharpening", value=True)
sharpen_strength = st.sidebar.slider("Eyewall Gradient Sharpening", 0.1, 0.8, 0.35, 0.05)
downscaler.diffusion_steps = diff_steps
downscaler.sharpening_strength = sharpen_strength

# Main Header
st.markdown('<div class="main-header">🌪️ METEORA — Atmospheric Cyclone Downscaler</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Stage Physics-Informed Pipeline: Centroid Tracking ➔ 12 km to 5 km Diffusion ➔ Flux Verification Stencil ➔ GIS Alerts</div>', unsafe_allow_html=True)

# Fetch Current Timestep Data
step_info = tracker.process_timestep(time_idx)
pressure_2d, lats, lons, timestamp_str = tracker.get_slice(time_idx)
centroid = step_info["centroid"]
min_p = step_info["min_pressure_hpa"]
intensity = step_info["intensity"]
bbox = step_info["bounding_box"]

# Execute Stage 2 Downscaling
downscaled_res = downscaler.downscale(pressure_2d, lats, lons, apply_diffusion=apply_diff)
hr_p = downscaled_res["hr_field"]
hr_lats = downscaled_res["hr_lats"]
hr_lons = downscaled_res["hr_lons"]

# Execute Stage 3 Physics Verification
phys_metrics = verifier.verify_field(hr_p, hr_lats, hr_lons)

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Central Pressure", f"{min_p:.1f} hPa", delta=f"{intensity['pressure_deficit_hpa']:.1f} deficit", delta_color="inverse")
with kpi2:
    st.metric("IMD Category", intensity["category"])
with kpi3:
    st.metric("Max Est. Wind", f"{intensity['vmax_knots']} kts", f"{intensity['vmax_kmh']} km/h")
with kpi4:
    st.metric("Centroid (Lat, Lon)", f"{centroid['lat']}°N, {centroid['lon']}°E")
with kpi5:
    score = phys_metrics["physics_consistency_score"]
    st.metric("Physics Stencil Score", f"{score}%", "Valid" if phys_metrics["is_physically_consistent"] else "Check", delta_color="normal")

# Severity Banner
alert_level = intensity["alert_level"]
badge_class = "alert-badge-red" if "RED" in alert_level else "alert-badge-orange"
st.markdown(
    f'<div><strong>Current Alert Level:</strong> <span class="{badge_class}">{alert_level} ({intensity["category"]})</span> &nbsp;|&nbsp; <strong>Valid Timestamp:</strong> {timestamp_str} UTC</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Stage 1: Trajectory & Bounding Box",
    "⚡ Stage 2: 12 km ➔ 5 km Downscaling",
    "🔬 Stage 3: Physics Verification Stencil",
    "🗺️ Stage 4: 5 km GIS Rings & Alerts"
])

# ----------------- TAB 1: TRACKING -----------------
with tab1:
    st.subheader("Stage 1: Vortex Centroid & Dynamic Bounding Box Extraction")
    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        # Trajectory map
        all_tracks = tracker.track_all()
        track_df = pd.DataFrame([
            {
                "time": t["timestamp"],
                "lat": t["centroid"]["lat"],
                "lon": t["centroid"]["lon"],
                "min_p": t["min_pressure_hpa"],
                "category": t["intensity"]["category"],
                "idx": t["time_index"]
            }
            for t in all_tracks
        ])

        fig_track = go.Figure()

        # Full track line
        fig_track.add_trace(go.Scattergeo(
            lon=track_df["lon"],
            lat=track_df["lat"],
            mode="lines+markers",
            line=dict(width=2, color="#00d2ff"),
            marker=dict(size=6, color="#00d2ff"),
            name="Observed Track"
        ))

        # Current centroid
        fig_track.add_trace(go.Scattergeo(
            lon=[centroid["lon"]],
            lat=[centroid["lat"]],
            mode="markers+text",
            marker=dict(size=14, color="#ff1744", symbol="star"),
            text=[f"Current ({min_p} hPa)"],
            textposition="top right",
            name="Current Eye"
        ))

        # Dynamic Bounding Box
        b_min_lon, b_min_lat, b_max_lon, b_max_lat = bbox
        fig_track.add_trace(go.Scattergeo(
            lon=[b_min_lon, b_max_lon, b_max_lon, b_min_lon, b_min_lon],
            lat=[b_min_lat, b_min_lat, b_max_lat, b_max_lat, b_min_lat],
            mode="lines",
            line=dict(width=2, color="#00e676", dash="dash"),
            name="5 km Patch Bounding Box"
        ))

        fig_track.update_geos(
            scope="asia",
            center=dict(lat=float(centroid["lat"]), lon=float(centroid["lon"])),
            projection_scale=6,
            showcoastlines=True,
            coastlinecolor="#666",
            showland=True,
            landcolor="#1e222d",
            showocean=True,
            oceancolor="#0e1117"
        )
        fig_track.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=460,
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
        )
        st.plotly_chart(fig_track, use_container_width=True)

    with col_t2:
        st.markdown("**Trajectory Pressure Deepening**")
        fig_p = px.line(
            track_df,
            x="time",
            y="min_p",
            markers=True,
            title="MSLP Minimum Evolution (hPa)",
            labels={"min_p": "Central Pressure (hPa)", "time": "Time (UTC)"},
            template="plotly_dark"
        )
        # Highlight current point
        fig_p.add_trace(go.Scatter(
            x=[track_df.iloc[time_idx]["time"]],
            y=[track_df.iloc[time_idx]["min_p"]],
            mode="markers",
            marker=dict(size=12, color="#ff1744"),
            name="Active Timestep"
        ))
        fig_p.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_p, use_container_width=True)

# ----------------- TAB 2: DOWNSCALING -----------------
with tab2:
    st.subheader("Stage 2: 12 km Coarse Input ➔ 5 km Diffusion-Enhanced Resolution")
    st.markdown("Physics-guided diffusion reconstruction enhances eyewall gradients and recovers mesoscale pressure contours.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Coarse Resolution Field (~12 km / GFS Grid)** — Shape: `{pressure_2d.shape}`")
        fig_coarse = go.Figure(data=go.Contour(
            z=pressure_2d,
            x=lons,
            y=lats,
            colorscale="Viridis",
            reversescale=True,
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="hPa")
        ))
        fig_coarse.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_coarse, use_container_width=True)

    with c2:
        st.markdown(f"**Super-Resolved Field (~5 km Diffusion Mesh)** — Shape: `{hr_p.shape}`")
        fig_hr = go.Figure(data=go.Contour(
            z=hr_p,
            x=hr_lons,
            y=hr_lats,
            colorscale="Viridis",
            reversescale=True,
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="hPa")
        ))
        fig_hr.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hr, use_container_width=True)

    # Gradient sharpening metric
    g1, g2, g3 = st.columns(3)
    g1.info(f"Coarse Max Gradient: **{downscaled_res['max_gradient_coarse']} hPa/deg**")
    g2.info(f"5 km Enhanced Max Gradient: **{downscaled_res['max_gradient_hr']} hPa/deg**")
    g3.success(f"Gradient Sharpening Factor: **{downscaled_res['gradient_enhancement_ratio']}x**")

# ----------------- TAB 3: PHYSICS VERIFICATION -----------------
with tab3:
    st.subheader("Stage 3: Mass Continuity & Moisture Flux Verification Stencil")
    st.markdown("Verifies physical validity of downscaled fields via horizontal divergence and moisture flux convergence (MFC).")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Derived Horizontal Wind Speed Field (m/s)**")
        fig_wind = go.Figure(data=go.Heatmap(
            z=phys_metrics["wind_speed_grid"],
            x=hr_lons,
            y=hr_lats,
            colorscale="Turbo",
            colorbar=dict(title="m/s")
        ))
        fig_wind.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_wind, use_container_width=True)

    with p2:
        st.markdown("**Moisture Flux Convergence $-\\nabla \\cdot \\mathbf{F}_q$ (kg/(m²s))**")
        fig_mfc = go.Figure(data=go.Heatmap(
            z=phys_metrics["mfc_grid"],
            x=hr_lons,
            y=hr_lats,
            colorscale="Blues",
            colorbar=dict(title="MFC")
        ))
        fig_mfc.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_mfc, use_container_width=True)

    st.markdown("### Stencil Verification Scorecard")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Mean Mass Divergence Residual", phys_metrics["mean_mass_divergence_residual"])
    sc2.metric("Core Moisture Flux Convergence", phys_metrics["core_moisture_flux_convergence"])
    sc3.metric("Max Gradient Wind Speed", f"{phys_metrics['max_wind_knots']} kts", f"{phys_metrics['max_wind_ms']} m/s")

# ----------------- TAB 4: GIS & ALERTS -----------------
with tab4:
    st.subheader("Stage 4: 5 km Concentric Polygon Rings & GIS Alert Broadcast")
    st.markdown("Concentric impact zones: 5 km Eye Core, 25 km Eyewall, 65 km Destructive, 150 km Gale, 250 km Outer Band.")

    gis_geojson = generate_cyclone_geojson(
        centroid_lat=centroid["lat"],
        centroid_lon=centroid["lon"],
        min_pressure_hpa=min_p,
        timestamp=timestamp_str,
        intensity_info=intensity,
        bbox=bbox
    )
    geojson_str = export_geojson_string(gis_geojson)

    gcol1, gcol2 = st.columns([3, 2])
    with gcol1:
        # Visualize the rings using scattergeo
        fig_rings = go.Figure()
        for feat in gis_geojson["features"]:
            geom = feat["geometry"]
            props = feat["properties"]
            if geom["type"] == "Point":
                fig_rings.add_trace(go.Scattergeo(
                    lon=[geom["coordinates"][0]],
                    lat=[geom["coordinates"][1]],
                    mode="markers+text",
                    marker=dict(size=14, color="#ff1744", symbol="star"),
                    name="Eye Center",
                    text=["Eye Center"],
                    textposition="bottom center"
                ))
            elif geom["type"] == "Polygon":
                ring_coords = geom["coordinates"][0]
                lons_r = [c[0] for c in ring_coords]
                lats_r = [c[1] for c in ring_coords]
                fig_rings.add_trace(go.Scattergeo(
                    lon=lons_r,
                    lat=lats_r,
                    mode="lines",
                    line=dict(
                        color=props.get("stroke", "#ff9800"),
                        width=props.get("stroke_width", 2),
                        dash="dash" if "bounding" in props.get("feature_type", "") else "solid"
                    ),
                    name=props.get("zone_name", props.get("description", "Zone"))
                ))

        fig_rings.update_geos(
            scope="asia",
            center=dict(lat=float(centroid["lat"]), lon=float(centroid["lon"])),
            projection_scale=10,
            showcoastlines=True,
            coastlinecolor="#888",
            showland=True,
            landcolor="#1c2128",
            showocean=True,
            oceancolor="#0b0e14"
        )
        fig_rings.update_layout(
            template="plotly_dark",
            height=460,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
        )
        st.plotly_chart(fig_rings, use_container_width=True)

    with gcol2:
        st.markdown("**GIS Export & Downstream Integration**")
        st.download_button(
            label="📥 Download GeoJSON (5 km Rings & BBox)",
            data=geojson_str,
            file_name=f"cyclone_biparjoy_rings_{time_idx:02d}.geojson",
            mime="application/geo+json"
        )

        alert_payload = {
            "alert_id": f"METEORA-ALERT-BIPARJOY-{time_idx:03d}",
            "timestamp": timestamp_str,
            "basin": "Arabian Sea",
            "storm": "Cyclone Biparjoy",
            "category": intensity["category"],
            "alert_level": intensity["alert_level"],
            "central_pressure_hpa": min_p,
            "max_wind_kmh": intensity["vmax_kmh"],
            "eye_coords": centroid,
            "bbox": bbox
        }
        st.download_button(
            label="🚨 Download JSON Alert Bulletin",
            data=json.dumps(alert_payload, indent=2),
            file_name=f"alert_biparjoy_bulletin_{time_idx:02d}.json",
            mime="application/json"
        )

        st.markdown("**Live Emergency JSON Alert Payload:**")
        st.json(alert_payload)
