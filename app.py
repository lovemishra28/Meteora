# app.py -- METEORA :: Cyclone Intelligence & Downscaling Console
# Production-style monitoring dashboard for the Biparjoy prototype.
# Powered by real ERA5 reanalysis (56 timesteps, 0.25 deg grid) + trained diffusion downscaler.

import os
import re
import json as _json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr

# Core imports
from core.data_loader import load_biparjoy_data, dataset_provenance
from core.tracker import AnomalyTracker
from core.downscaler import DownscalerEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meteora Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme system : shared layout, two palettes, single #D67557 accent, no gradients ──
# Streamlit's .streamlit/config.toml theme is static at load, so the runtime
# dark/light switch is driven entirely by injected CSS variables + theme-aware
# Plotly/folium colours. The accent (#D67557) is constant across both themes.
_DARK = {
    "bg": "#000000", "panel": "#0A0A0A", "panel2": "#0E0E0E",
    "border": "#1C1C1C", "border2": "#2A2A2A",
    "accent": "#D67557", "accent_soft": "rgba(214,117,87,0.12)", "accent_line": "rgba(214,117,87,0.45)",
    "tx": "#F2F2F2", "tx2": "#9A9A9A", "tx3": "#5E5E5E",
    "side": "#050505", "scroll": "#242424", "logrow": "#131313", "loghover": "#0D0D0D",
    "chart_paper": "#000000", "chart_grid": "#171717", "chart_font": "#CFCFCF",
    "fill": "rgba(214,117,87,0.10)", "thresh2": "#333333",
    "seq": [[0.0, "#000000"], [0.32, "#3a2318"], [0.62, "#a24e34"], [0.82, "#D67557"], [1.0, "#f0b89f"]],
    "div": [[0.0, "#c8c8c8"], [0.5, "#000000"], [1.0, "#D67557"]],
    "bar": ["#5E5E5E", "#8A8A8A"],
    "tiles": "OpenStreetMap",
    "foot_tx": "#3A3A3A", "foot_border": "#141414",
}
_LIGHT = {
    "bg": "#F4F2EF", "panel": "#FFFFFF", "panel2": "#FAF7F3",
    "border": "#E7E1DA", "border2": "#D8D0C6",
    "accent": "#D67557", "accent_soft": "rgba(214,117,87,0.14)", "accent_line": "rgba(214,117,87,0.55)",
    "tx": "#201C18", "tx2": "#6E655C", "tx3": "#A79D92",
    "side": "#FFFFFF", "scroll": "#D8D0C6", "logrow": "#EFEAE3", "loghover": "#F7F2ED",
    "chart_paper": "#FFFFFF", "chart_grid": "#ECE6DF", "chart_font": "#4A433B",
    "fill": "rgba(214,117,87,0.13)", "thresh2": "#CFC7BC",
    "seq": [[0.0, "#FBF7F4"], [0.30, "#E7C4B4"], [0.60, "#E19E82"], [0.82, "#D67557"], [1.0, "#9A4227"]],
    "div": [[0.0, "#8C8378"], [0.5, "#FBF7F4"], [1.0, "#D67557"]],
    "bar": ["#CBC3B8", "#A79D92"],
    "tiles": "OpenStreetMap",
    "foot_tx": "#B3A99E", "foot_border": "#E7E1DA",
}
THEMES = {"dark": _DARK, "light": _LIGHT}

st.session_state.setdefault("dark_mode", True)
IS_DARK = st.session_state["dark_mode"]
T = THEMES["dark" if IS_DARK else "light"]

# Aliases used by Plotly / folium (CSS vars can't reach those renderers)
ACCENT  = T["accent"]
INK     = T["chart_paper"]
PANEL   = T["panel"]
GRID    = T["chart_grid"]
TX      = T["chart_font"]
C_FILL  = T["fill"]
THRESH2 = T["thresh2"]
TX2H    = T["tx2"]
TX3H    = T["tx3"]
BORDER2 = T["border2"]
SEQ     = T["seq"]
DIV     = T["div"]
BAR     = T["bar"]
TILES   = T["tiles"]

# ── Inject palette variables (theme-dependent) ────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
:root {{
  --bg:{T['bg']}; --panel:{T['panel']}; --panel2:{T['panel2']};
  --border:{T['border']}; --border2:{T['border2']};
  --accent:{T['accent']}; --accent-soft:{T['accent_soft']}; --accent-line:{T['accent_line']};
  --tx:{T['tx']}; --tx2:{T['tx2']}; --tx3:{T['tx3']};
  --side:{T['side']}; --scroll:{T['scroll']}; --logrow:{T['logrow']}; --loghover:{T['loghover']};
}}
</style>""", unsafe_allow_html=True)

# ── Static styling (references the palette variables above) ───────────────────
st.markdown("""
<style>
.stApp{ background:var(--bg); color:var(--tx);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
.block-container{ padding-top:1.1rem; padding-bottom:2.2rem; max-width:1520px; }

/* strip Streamlit chrome for a product feel */
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none;} [data-testid="stDecoration"]{display:none;}
[data-testid="stStatusWidget"]{display:none;}
header[data-testid="stHeader"]{background:transparent;height:0;}

/* typography */
html,body,.stApp,p,span,div,label{ font-size:15px; }
.stMarkdown p{ font-size:0.98rem; line-height:1.55; color:var(--tx2); }
.mono{ font-family:'JetBrains Mono',ui-monospace,monospace; }

/* sidebar */
section[data-testid="stSidebar"]{ background:var(--side) !important; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] *{ color:var(--tx); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }

/* brand */
.brand{ display:flex; align-items:center; gap:14px; margin-bottom:2px; }
.brand-mark{ width:40px;height:40px;border:1.5px solid var(--accent);border-radius:10px;
  display:flex;align-items:center;justify-content:center; flex:0 0 auto; }
.brand-name{ font-size:1.55rem; font-weight:800; letter-spacing:0.16em; color:var(--tx); line-height:1; }
.brand-tag{ font-size:0.7rem; letter-spacing:0.3em; text-transform:uppercase; color:var(--tx3); margin-top:5px; }

/* status bar */
.statusbar{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 2px; }
.chip{ display:flex;align-items:center;gap:9px; background:var(--panel);
  border:1px solid var(--border); border-radius:9px; padding:8px 14px;
  font-size:0.76rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--tx3); font-weight:600; }
.chip b{ color:var(--tx); font-weight:700; letter-spacing:0.03em; text-transform:none; font-size:0.82rem; }
.chip .dot{ width:8px;height:8px;border-radius:50%; background:var(--accent); flex:0 0 auto; }
.chip .dot.pulse{ animation:pulse 1.9s ease-in-out infinite; }
.chip.mute .dot{ background:var(--tx3); }

@keyframes pulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(214,117,87,0.6);} 50%{ box-shadow:0 0 0 6px rgba(214,117,87,0);} }
@keyframes rise{ from{opacity:0; transform:translateY(9px);} to{opacity:1; transform:translateY(0);} }

/* section header */
.sec{ font-size:0.76rem; letter-spacing:0.17em; text-transform:uppercase;
  color:var(--tx2); font-weight:700; margin:8px 0 12px; display:flex; align-items:center; gap:10px; }
.sec::before{ content:""; width:3px; height:15px; background:var(--accent); border-radius:2px; }
.sec .n{ color:var(--tx3); font-weight:600; letter-spacing:0.08em; }

/* KPI grid */
.kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:2px 0; }
.kpi{ background:var(--panel); border:1px solid var(--border); border-radius:13px; padding:17px 19px;
  transition:border-color .18s ease, transform .18s ease; animation:rise .4s ease both; }
.kpi:hover{ border-color:var(--border2); transform:translateY(-2px); }
.kpi.hot{ border-color:var(--accent-line); }
.kpi-l{ font-size:0.68rem; letter-spacing:0.13em; text-transform:uppercase; color:var(--tx3); font-weight:700; }
.kpi-v{ font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; color:var(--tx); margin-top:7px; line-height:1.02; }
.kpi.hot .kpi-v{ color:var(--accent); }
.kpi-u{ font-size:0.8rem; color:var(--tx3); margin-left:5px; font-weight:500; }
.kpi-s{ font-size:0.74rem; color:var(--tx2); margin-top:7px; letter-spacing:0.02em; }
.kpi-s .a{ color:var(--accent); }

/* alert strip */
.alert{ display:flex; align-items:center; gap:16px; border-radius:13px; padding:15px 20px; margin:14px 0 4px;
  background:var(--panel); border:1px solid var(--border); border-left:4px solid var(--accent);
  animation:rise .4s ease both; }
.alert.calm{ border-left-color:var(--tx3); }
.badge{ font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700; letter-spacing:0.11em;
  padding:6px 13px; border-radius:7px; flex:0 0 auto; }
.badge.crit{ background:var(--accent); color:#0a0a0a; }
.badge.high{ background:transparent; color:var(--accent); border:1px solid var(--accent); }
.badge.mod{ background:transparent; color:var(--tx); border:1px solid var(--border2); }
.badge.adv{ background:transparent; color:var(--tx2); border:1px solid var(--border); }
.alert-cat{ font-size:1.12rem; font-weight:700; color:var(--tx); letter-spacing:0.01em; }
.alert-meta{ font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:var(--tx2); margin-left:auto; text-align:right; }

/* generic panel */
.panel{ background:var(--panel); border:1px solid var(--border); border-radius:13px; padding:6px 4px 2px; }

/* log table */
.logwrap{ max-height:236px; overflow-y:auto; border:1px solid var(--border); border-radius:12px; background:var(--panel); }
.log{ font-family:'JetBrains Mono',monospace; font-size:0.8rem; width:100%; border-collapse:collapse; }
.log th{ position:sticky; top:0; background:var(--panel); text-align:left; color:var(--tx3); font-weight:700;
  font-size:0.66rem; letter-spacing:0.12em; text-transform:uppercase; padding:10px 14px; border-bottom:1px solid var(--border); }
.log td{ padding:9px 14px; border-bottom:1px solid var(--logrow); color:var(--tx2); white-space:nowrap; }
.log tr:hover td{ background:var(--loghover); color:var(--tx); }
.log .a{ color:var(--accent); font-weight:600; }
.log .now td{ background:var(--accent-soft); }
.log .now td:first-child{ box-shadow:inset 3px 0 0 var(--accent); }

/* machine-readable alert block */
.jsonbox{ background:var(--panel2); border:1px solid var(--border); border-radius:12px;
  padding:16px 18px; font-family:'JetBrains Mono',monospace; font-size:0.8rem; line-height:1.55;
  color:var(--tx2); white-space:pre; overflow:auto; max-height:436px; margin:2px 0; }
.jsonbox .k{ color:var(--accent); }
.note-inline{ background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--accent-line);
  border-radius:10px; padding:12px 16px; font-size:0.85rem; color:var(--tx2); margin:6px 0; }

/* tabs */
button[data-baseweb="tab"]{ font-size:0.9rem !important; font-weight:600 !important; letter-spacing:0.04em; color:var(--tx2) !important; padding:11px 2px !important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--accent) !important; }
[data-baseweb="tab-list"]{ gap:30px; border-bottom:1px solid var(--border) !important; }
[data-baseweb="tab-highlight"]{ background:var(--accent) !important; height:2px !important; }
[data-baseweb="tab-border"]{ background:transparent !important; }

/* buttons */
.stButton>button{ background:transparent; color:var(--accent); border:1px solid var(--accent-line);
  border-radius:9px; font-weight:600; font-size:0.84rem; letter-spacing:0.03em; transition:all .16s ease; width:100%; }
.stButton>button:hover{ background:var(--accent-soft); border-color:var(--accent); color:var(--accent); }

/* native widgets follow the active theme (config.toml can't switch at runtime) */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{ color:var(--tx2) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{ color:var(--tx3) !important; }
.stSelectbox > div > div { background:var(--panel) !important; border-color:var(--border) !important; }
.stSelectbox > div > div > div { background:var(--panel) !important; }
.stSelectbox * { color:var(--tx) !important; }
div[data-baseweb="select"] svg { fill:var(--tx2) !important; }
div[data-baseweb="popover"] ul { background:var(--panel) !important; border:1px solid var(--border) !important; }
div[data-baseweb="popover"] li { background:var(--panel) !important; color:var(--tx) !important; }
div[data-baseweb="popover"] li:hover { background:var(--accent-soft) !important; color:var(--tx) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{ color:var(--tx3) !important; background:transparent !important; }
[data-testid="stThumbValue"]{ color:var(--accent) !important; }
[data-testid="stRadio"] label, [data-testid="stRadio"] label p{ color:var(--tx2) !important; }

/* theme toggle: keep it compact and right-aligned in the header */
.themebar [data-testid="stWidgetLabel"]{ margin-bottom:0 !important; }
.themebar{ display:flex; justify-content:flex-end; padding-top:8px; }

/* misc */
hr{ border-color:var(--border) !important; margin:15px 0 !important; }
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:var(--scroll); border-radius:5px; }
::-webkit-scrollbar-thumb:hover{ background:var(--accent); }
</style>
""", unsafe_allow_html=True)

BADGE = {"CRITICAL": "crit", "HIGH": "high", "MODERATE": "mod", "ADVISORY": "adv"}


def alert_hex(level: str) -> str:
    return ACCENT if level in ("CRITICAL", "HIGH") else TX3H


def style_fig(fig, height=250, title=None, legend=False):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TX), x=0.01) if title else None,
        paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(color=TX, family="Inter, sans-serif", size=13),
        margin=dict(l=10, r=12, t=(38 if title else 8), b=8), height=height,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        showlegend=legend,
        legend=dict(bgcolor=PANEL, bordercolor=BORDER2, borderwidth=1,
                    font=dict(size=11, color=TX)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=ACCENT, font=dict(color=TX)),
    )
    return fig


def kpi(label, value, unit="", sub="", hot=False):
    cls = "kpi hot" if hot else "kpi"
    u = f'<span class="kpi-u">{unit}</span>' if unit else ""
    s = f'<div class="kpi-s">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="kpi-l">{label}</div><div class="kpi-v">{value}{u}</div>{s}</div>'


def chip(label, value, live=False, mute=False):
    dot = f'<span class="dot {"pulse" if live else ""}"></span>'
    cls = "chip mute" if mute else "chip"
    return f'<div class="{cls}">{dot}{label}&nbsp;<b>{value}</b></div>'


def json_block(obj):
    """Render a machine-readable alert as a themed code block (keys accented)."""
    txt = _json.dumps(obj, indent=2)
    txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    txt = re.sub(r'("(?:[^"\\]|\\.)*")(\s*:)', r'<span class="k">\1</span>\2', txt)
    st.markdown(f'<pre class="jsonbox">{txt}</pre>', unsafe_allow_html=True)


# ── System init (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising engines…")
def init_system():
    return AnomalyTracker(), DownscalerEngine()


@st.cache_data(show_spinner="Computing ERA5 trajectory…")
def compute_trajectory(_tracker):
    return _tracker.get_full_trajectory()


tracker, downscaler = init_system()
provenance = dataset_provenance(tracker.ds)
mode = downscaler.mode

# Live feed status (best-effort)
feed_src, feed_fresh, feed_time = "none", False, "—"
try:
    from core.live_ingestion import LiveIngestionPipeline
    _s = LiveIngestionPipeline().get_status()
    feed_src = _s.get("source", "none")
    feed_fresh = bool(_s.get("fresh", False))
    _ft = _s.get("fetched_at")
    feed_time = _ft[:16].replace("T", " ") if _ft else "—"
except Exception:
    pass


# ── Sidebar : control panel ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">'
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#D67557" '
        'stroke-width="1.7" stroke-linecap="round"><path d="M12 12c0-3 3-4 5-3M12 12c0 3-3 4-5 3'
        'M12 12c-3 0-4-3-3-5M12 12c3 0 4 3 3 5"/><circle cx="12" cy="12" r="1.5" fill="#D67557" '
        'stroke="none"/></svg></div><div><div class="brand-name">METEORA</div>'
        '<div class="brand-tag">Cyclone Console</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Controls</div>', unsafe_allow_html=True)
    n_times = len(tracker.times)
    time_idx = st.slider("Forecast timestep", 0, n_times - 1,
                         value=min(30, n_times - 1), step=1,
                         help=f"{n_times} ERA5 six-hourly steps across the cyclone lifecycle")
    ts_label = pd.to_datetime(str(tracker.times[time_idx])).strftime("%d %b %Y · %H:%M UTC")
    st.markdown(
        f'<div class="chip" style="width:100%;justify-content:center;margin:-4px 0 6px">'
        f'<span class="dot"></span>VALID&nbsp;<b>{ts_label}</b></div>', unsafe_allow_html=True)

    variable = st.selectbox(
        "Downscaling field",
        ["u10", "v10", "mslp", "precipitation"],
        format_func=lambda x: {"u10": "U-Wind (km/h)", "v10": "V-Wind (km/h)",
                               "mslp": "Pressure (hPa)", "precipitation": "Rainfall (mm/h)"}[x],
    )

    st.markdown('<div class="sec" style="margin-top:22px">System</div>', unsafe_allow_html=True)
    _src_label = ("ERA5 Reanalysis" if "ERA5" in provenance
                  else "Live Feed" if "LIVE" in provenance else "Simulated")
    _model_label = ("Diffusion · active" if mode == "PGDM_REAL"
                    else "Checkpoint loaded" if mode == "PGDM_UNTRAINED" else "Simulation mode")
    st.markdown(
        chip("Data source", _src_label, mute=("ERA5" not in provenance and "LIVE" not in provenance)) +
        chip("Downscaler", _model_label, mute=(mode != "PGDM_REAL")) +
        chip("Live feed", (feed_src.replace("LIVE_", "").title() if feed_fresh else "stale"),
             live=feed_fresh, mute=not feed_fresh),
        unsafe_allow_html=True,
    )
    st.caption(f"Feed updated {feed_time} UTC" if feed_fresh else "Live ingest idle")
    if st.button("Refresh live feed"):
        try:
            from core.live_ingestion import run_live_ingest
            with st.spinner("Fetching (Open-Meteo → GFS → ERA5)…"):
                run_live_ingest(force=True)
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.caption(f"Ingest unavailable: {e}")


# ── Compute current state ─────────────────────────────────────────────────────
storm      = tracker.detect_storm_center(time_idx)
patch      = tracker.crop_anomaly_subgrid(time_idx)
metrics    = downscaler.process_anomaly_patch(patch, variable=variable)
trajectory = compute_trajectory(tracker)

c_lat, c_lon = storm["centroid"]
alert_level  = storm["alert_level"]
alert_color  = alert_hex(alert_level)
b            = storm["bbox"]


# ── Header + theme toggle + status bar ────────────────────────────────────────
_brand_html = (
    '<div class="brand"><div class="brand-mark">'
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D67557" '
    'stroke-width="1.7" stroke-linecap="round"><path d="M12 12c0-3 3-4 5-3M12 12c0 3-3 4-5 3'
    'M12 12c-3 0-4-3-3-5M12 12c3 0 4 3 3 5"/><circle cx="12" cy="12" r="1.5" fill="#D67557" '
    'stroke="none"/></svg></div><div><div class="brand-name">METEORA</div>'
    '<div class="brand-tag">Cyclone Intelligence &middot; Arabian Sea Basin</div></div></div>'
)
h_left, h_right = st.columns([6, 1], gap="small")
with h_left:
    st.markdown(_brand_html, unsafe_allow_html=True)
with h_right:
    st.markdown('<div class="themebar">', unsafe_allow_html=True)
    st.toggle("Dark mode", key="dark_mode")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="statusbar">'
    + chip("System", "Online", live=True)
    + chip("Event", "Biparjoy · 2023")
    + chip("Source", _src_label, mute=("ERA5" not in provenance and "LIVE" not in provenance))
    + chip("Downscaler", ("Diffusion" if mode == "PGDM_REAL" else "Simulation"), mute=(mode != "PGDM_REAL"))
    + chip("Grid", "0.25° · ~31 km")
    + chip("Steps", str(n_times))
    + '</div>',
    unsafe_allow_html=True,
)

# Alert strip
_calm = alert_level in ("MODERATE", "ADVISORY")
st.markdown(
    f'<div class="alert {"calm" if _calm else ""}">'
    f'<span class="badge {BADGE.get(alert_level, "adv")}">{alert_level}</span>'
    f'<span class="alert-cat">{storm["category"]}</span>'
    f'<span class="alert-meta">{c_lat:.2f}°N&nbsp; {c_lon:.2f}°E'
    f'<br>box {b["lat_min"]}–{b["lat_max"]}N · {b["lon_min"]}–{b["lon_max"]}E</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# KPI grid
_deficit = round(1010 - storm["min_mslp_hpa"], 1)
st.markdown(
    '<div class="kpi-grid">'
    + kpi("Valid time", ts_label.split(" · ")[1].replace(" UTC", ""), "UTC",
          sub=ts_label.split(" · ")[0])
    + kpi("Min pressure", f"{storm['min_mslp_hpa']:.0f}", "hPa",
          sub=f'<span class="a">▼ {_deficit}</span> hPa deficit', hot=True)
    + kpi("Peak wind", f"{storm['max_wind_kmh']:.0f}", "km/h",
          sub=f"{storm['max_wind_kmh'] / 1.852:.0f} kt sustained")
    + kpi("Peak rainfall", f"{storm['max_precip_mmh']:.1f}", "mm/h",
          sub="patch maximum")
    + kpi("Storm centre", f"{c_lat:.2f}", "°N",
          sub=f"{c_lon:.2f} °E")
    + '</div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── Command centre : map + intensity ──────────────────────────────────────────
map_col, chart_col = st.columns([1.5, 1], gap="medium")

with map_col:
    st.markdown('<div class="sec">Live track &amp; impact zone <span class="n">Stage 1</span></div>',
                unsafe_allow_html=True)

    m = folium.Map(location=[c_lat, c_lon], zoom_start=6,
                   tiles=TILES, control_scale=True)

    pts = [(t["centroid"][0], t["centroid"][1]) for t in trajectory]
    folium.PolyLine(pts, color=ACCENT, weight=2.5, opacity=0.85,
                    tooltip="ERA5 storm track").add_to(m)

    for i, t in enumerate(trajectory[::2]):
        clat2, clon2 = t["centroid"]
        col2 = alert_hex(t["alert_level"])
        folium.CircleMarker(
            location=[clat2, clon2], radius=3, color=col2, weight=1,
            fill=True, fill_color=col2, fill_opacity=0.85,
            tooltip=f"T={i*2}: {t['timestamp']} · {t['min_mslp_hpa']} hPa",
        ).add_to(m)

    folium.Rectangle(
        bounds=[[b["lat_min"], b["lon_min"]], [b["lat_max"], b["lon_max"]]],
        color=ACCENT, weight=2, fill=True, fill_color=ACCENT, fill_opacity=0.06,
        dash_array="6,6", tooltip="Tracking bounding box",
    ).add_to(m)

    folium.Circle(location=[c_lat, c_lon], radius=5000, color=ACCENT, weight=1.5,
                  fill=True, fill_color=ACCENT, fill_opacity=0.30,
                  tooltip="5 km impact core").add_to(m)

    folium.CircleMarker(
        location=[c_lat, c_lon], radius=8, color=ACCENT, weight=2.5,
        fill=True, fill_color=ACCENT, fill_opacity=0.9,
        popup=folium.Popup(
            f"<b>{storm['category']}</b><br>{storm['timestamp']}<br>"
            f"MSLP {storm['min_mslp_hpa']} hPa · Wind {storm['max_wind_kmh']} km/h",
            max_width=250),
        tooltip="Current centre",
    ).add_to(m)

    st_folium(m, width=None, height=468, key=f"map_{time_idx}_{'d' if IS_DARK else 'l'}")

with chart_col:
    st.markdown('<div class="sec">Intensity timeline</div>', unsafe_allow_html=True)

    df_traj = pd.DataFrame([{
        "time": pd.to_datetime(t["timestamp"].replace(" UTC", "")),
        "mslp": t["min_mslp_hpa"],
        "wind": t["max_wind_kmh"],
    } for t in trajectory])
    cur_time = df_traj["time"].iloc[time_idx]

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=df_traj["time"], y=df_traj["mslp"], mode="lines",
        line=dict(color=ACCENT, width=2.4), fill="tozeroy",
        fillcolor=C_FILL, name="MSLP",
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} hPa<extra></extra>"))
    fig_p.add_vline(x=cur_time, line_color=ACCENT, line_width=1.6, line_dash="dot")
    fig_p.add_hline(y=970, line_color="rgba(214,117,87,0.55)", line_dash="dash",
                    annotation_text="970 ESCS", annotation_position="top left",
                    annotation_font=dict(size=10, color=TX2H))
    fig_p.add_hline(y=985, line_color=THRESH2, line_dash="dash",
                    annotation_text="985 VSCS", annotation_position="bottom left",
                    annotation_font=dict(size=10, color=TX3H))
    fig_p.update_yaxes(range=[df_traj["mslp"].min() - 6, df_traj["mslp"].max() + 6])
    st.plotly_chart(style_fig(fig_p, height=228, title="Minimum central pressure · hPa"),
                    use_container_width=True, config={"displayModeBar": False})

    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(
        x=df_traj["time"], y=df_traj["wind"], mode="lines",
        line=dict(color=ACCENT, width=2.4), fill="tozeroy",
        fillcolor=C_FILL, name="Wind",
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f} km/h<extra></extra>"))
    fig_w.add_vline(x=cur_time, line_color=ACCENT, line_width=1.6, line_dash="dot")
    st.plotly_chart(style_fig(fig_w, height=214, title="Maximum sustained wind · km/h"),
                    use_container_width=True, config={"displayModeBar": False})


# ── Track log (density + realism) ─────────────────────────────────────────────
st.markdown('<div class="sec" style="margin-top:6px">Track log <span class="n">ERA5 reanalysis</span></div>',
            unsafe_allow_html=True)

rows = ""
for i, t in enumerate(trajectory):
    if i % 4 != 0 and i != time_idx:
        continue
    hot = "now" if i == time_idx else ""
    ac = "a" if t["alert_level"] in ("CRITICAL", "HIGH") else ""
    cl2, cn2 = t["centroid"]
    rows += (
        f'<tr class="{hot}">'
        f'<td>T+{i*6:03d}h</td>'
        f'<td>{t["timestamp"].replace(" UTC","")}</td>'
        f'<td>{cl2:.2f}N {cn2:.2f}E</td>'
        f'<td class="{ac}">{t["min_mslp_hpa"]:.0f} hPa</td>'
        f'<td>{t["max_wind_kmh"]:.0f} km/h</td>'
        f'<td class="{ac}">{t["alert_level"]}</td>'
        f'</tr>'
    )
st.markdown(
    '<div class="logwrap"><table class="log"><thead><tr>'
    '<th>Lead</th><th>Timestamp</th><th>Centre</th><th>Pressure</th><th>Wind</th><th>Alert</th>'
    '</tr></thead><tbody>' + rows + '</tbody></table></div>',
    unsafe_allow_html=True,
)


# ── Detail tabs ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
tab_ds, tab_field, tab_phys = st.tabs([
    "Downscaling engine", "Field analysis", "Physics & alert",
])

field_label = {"u10": "U-Wind (km/h)", "v10": "V-Wind (km/h)",
               "mslp": "Pressure (hPa)", "precipitation": "Rainfall (mm/h)"}[variable]


def field_scale(name):
    if name == "mslp":
        return SEQ, True
    if name in ("u10", "v10"):
        return DIV, False
    return SEQ, False


# ── TAB : Downscaling ─────────────────────────────────────────────────────────
with tab_ds:
    st.markdown('<div class="sec">12 km &rarr; 5 km reconstruction <span class="n">Stage 2 · '
                + field_label + '</span></div>', unsafe_allow_html=True)

    cnn_drop = metrics["original_peak"] - metrics["cnn_peak"]
    st.markdown(
        '<div class="kpi-grid">'
        + kpi("Coarse peak", f"{metrics['original_peak']:.1f}", sub="12 km input field")
        + kpi("CNN peak", f"{metrics['cnn_peak']:.1f}",
              sub=f'<span class="a">−{abs(cnn_drop):.1f}</span> spectral smoothing')
        + kpi("Diffusion peak", f"{metrics['diffusion_peak']:.1f}",
              sub="extreme preserved", hot=True)
        + '</div>',
        unsafe_allow_html=True,
    )

    g = metrics["grids"]
    cscale, rev = field_scale(variable)
    vmin = float(min(g["coarse_12km"].min(), g["cnn_5km"].min(), g["diffusion_5km"].min()))
    vmax = float(max(g["coarse_12km"].max(), g["cnn_5km"].max(), g["diffusion_5km"].max()))

    def heatmap(data, title):
        fig = px.imshow(data, color_continuous_scale=cscale, zmin=vmin, zmax=vmax, aspect="auto")
        if rev:
            fig.update_coloraxes(reversescale=True)
        fig.update_layout(coloraxis_showscale=False)
        style_fig(fig, height=300, title=title)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        return fig

    h1, h2, h3 = st.columns(3, gap="small")
    h1.plotly_chart(heatmap(g["coarse_12km"], "Coarse · 12 km"),
                    use_container_width=True, config={"displayModeBar": False})
    h2.plotly_chart(heatmap(g["cnn_5km"], "Standard CNN · 5 km"),
                    use_container_width=True, config={"displayModeBar": False})
    h3.plotly_chart(heatmap(g["diffusion_5km"], "Meteora diffusion · 5 km"),
                    use_container_width=True, config={"displayModeBar": False})

    fig_bar = go.Figure(go.Bar(
        x=["Coarse 12 km", "Standard CNN", "Meteora diffusion"],
        y=[metrics["original_peak"], metrics["cnn_peak"], metrics["diffusion_peak"]],
        marker_color=[BAR[0], BAR[1], ACCENT],
        text=[f"{v:.1f}" for v in [metrics["original_peak"], metrics["cnn_peak"], metrics["diffusion_peak"]]],
        textposition="outside", textfont=dict(color=TX, size=13),
        hovertemplate="%{x}<br>peak %{y:.2f}<extra></extra>"))
    st.plotly_chart(style_fig(fig_bar, height=300, title=f"Peak retention · {field_label}"),
                    use_container_width=True, config={"displayModeBar": False})


# ── TAB : Field analysis ──────────────────────────────────────────────────────
with tab_field:
    st.markdown('<div class="sec">Full-grid ERA5 field</div>', unsafe_allow_html=True)

    field_choice = st.radio(
        "Field", ["mslp", "u10", "v10", "precipitation"], horizontal=True, label_visibility="collapsed",
        format_func=lambda x: {"mslp": "Pressure", "u10": "U-Wind", "v10": "V-Wind",
                               "precipitation": "Rainfall"}[x])

    slice_t = tracker.ds.isel(time=time_idx)
    lats2, lons2 = slice_t.lat.values, slice_t.lon.values
    data_2d = slice_t[field_choice].values
    units_map = {"mslp": "hPa", "u10": "km/h", "v10": "km/h", "precipitation": "mm/h"}
    cscale2, rev2 = field_scale(field_choice)

    fig_full = go.Figure(go.Contour(
        z=data_2d, x=lons2, y=lats2, colorscale=cscale2, reversescale=rev2,
        contours=dict(showlabels=True, labelfont=dict(color=TX, size=9)),
        line_width=0.5,
        colorbar=dict(title=dict(text=units_map[field_choice], font=dict(color=TX)),
                      tickfont=dict(color=TX), outlinecolor=GRID, thickness=14, len=0.85)))
    fig_full.add_trace(go.Scatter(
        x=[c_lon], y=[c_lat], mode="markers+text",
        marker=dict(symbol="x", size=13, color=ACCENT, line=dict(width=2, color=PANEL)),
        text=["  eye"], textposition="middle right", textfont=dict(color=TX, size=11),
        name="Centre"))
    bx = [b["lon_min"], b["lon_max"], b["lon_max"], b["lon_min"], b["lon_min"]]
    by = [b["lat_min"], b["lat_min"], b["lat_max"], b["lat_max"], b["lat_min"]]
    fig_full.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                  line=dict(color=ACCENT, width=1.8, dash="dash"), name="Bounding box"))
    style_fig(fig_full, height=560, title=f"{field_choice.upper()} · {ts_label}", legend=True)
    fig_full.update_xaxes(title="Longitude °E")
    fig_full.update_yaxes(title="Latitude °N")
    st.plotly_chart(fig_full, use_container_width=True, config={"displayModeBar": False})


# ── TAB : Physics & alert ─────────────────────────────────────────────────────
with tab_phys:
    p_left, p_right = st.columns([1.15, 1], gap="medium")

    with p_left:
        st.markdown('<div class="sec">Flow balance <span class="n">Stage 3</span></div>',
                    unsafe_allow_html=True)

        u_p = patch["u10"].values if "u10" in patch else np.zeros((5, 5))
        v_p = patch["v10"].values if "v10" in patch else np.zeros((5, 5))
        lats_p, lons_p = patch.lat.values, patch.lon.values

        if u_p.ndim == 2 and u_p.shape[0] > 1 and u_p.shape[1] > 1:
            dy_m = np.gradient(lats_p) * 111000
            dx_m = np.gradient(lons_p) * 111000 * np.cos(np.radians(np.mean(lats_p)))
            div = np.gradient(v_p, dy_m, axis=0) + np.gradient(u_p, dx_m, axis=1)
            div_rms = float(np.sqrt(np.mean(div ** 2)))
            wind_max = float(np.max(np.sqrt(u_p ** 2 + v_p ** 2)))
            press_min = float(patch["mslp"].min()) if "mslp" in patch else 0.0

            st.markdown(
                '<div class="kpi-grid">'
                + kpi("Divergence RMS", f"{div_rms:.4f}", "s⁻¹", sub="near-zero = balanced", hot=True)
                + kpi("Patch max wind", f"{wind_max:.0f}", "km/h")
                + kpi("Patch min pressure", f"{press_min:.0f}", "hPa")
                + '</div>',
                unsafe_allow_html=True,
            )

            fig_div = px.imshow(div, color_continuous_scale=DIV, aspect="auto",
                                zmin=-abs(div).max(), zmax=abs(div).max())
            fig_div.update_layout(coloraxis_colorbar=dict(title="s⁻¹", tickfont=dict(color=TX),
                                  outlinecolor=GRID, thickness=13, len=0.8))
            style_fig(fig_div, height=330, title="Horizontal wind divergence")
            fig_div.update_xaxes(showticklabels=False, showgrid=False)
            fig_div.update_yaxes(showticklabels=False, showgrid=False)
            st.plotly_chart(fig_div, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="note-inline">Patch too small for divergence at this timestep.</div>',
                        unsafe_allow_html=True)

    with p_right:
        st.markdown('<div class="sec">Machine-readable alert <span class="n">CAP / JSON</span></div>',
                    unsafe_allow_html=True)
        json_block({
            "alert_id":    f"BIPARJOY-ERA5-T{time_idx:03d}",
            "timestamp":   storm["timestamp"],
            "severity":    storm["alert_level"],
            "category":    storm["category"],
            "data_source": provenance,
            "impact_zone": {
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "bbox":         storm["bbox"],
                "resolution":   "5 km",
            },
            "peak_metrics": {
                "min_pressure_hpa": storm["min_mslp_hpa"],
                "max_wind_kmh":     storm["max_wind_kmh"],
                "max_rainfall_mmh": storm["max_precip_mmh"],
                "diffusion_peak":   metrics["diffusion_peak"],
            },
            "protocol": (
                "Deploy response assets to centroid; issue red alert within 200 km."
                if storm["alert_level"] in ("CRITICAL", "HIGH")
                else "Monitor; next update in 6 h."
            ),
        })


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;color:{T["foot_tx"]};font-size:0.75rem;'
    f'letter-spacing:0.08em;padding:20px 0 6px;border-top:1px solid {T["foot_border"]};margin-top:22px">'
    'METEORA · ERA5 reanalysis (ECMWF / Copernicus CDS) · SIH 2025</div>',
    unsafe_allow_html=True,
)
