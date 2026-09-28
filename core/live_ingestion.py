# core/live_ingestion.py
"""
Meteora Live Data Ingestion Pipeline
=====================================
Automatically fetches the latest available weather forecast/reanalysis data
from public meteorological data sources and normalises it into the Meteora
canonical schema for real-time dashboard use.

Data sources (in priority order):
  1. NCMRWF GFS OpenDAP server  -- operational forecast (NWP, free, no auth)
  2. Open-Meteo API              -- free JSON API (no key needed), ERA5-back
  3. ERA5 CDS fallback           -- if CDS credentials are configured
  4. Static biparjoy_real.nc     -- always-available last resort

Usage (CLI):
    python -c "from core.live_ingestion import run_live_ingest; run_live_ingest()"

Usage (API endpoint):
    POST /api/v1/ingest      -> triggers async fetch, returns job status
    GET  /api/v1/ingest/status -> returns latest cached dataset info

Usage (programmatic):
    from core.live_ingestion import LiveIngestionPipeline
    pipeline = LiveIngestionPipeline()
    ds = pipeline.fetch()           # returns xr.Dataset in canonical schema
    ds = pipeline.get_latest()      # returns cached dataset (no network call)
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

# Project root paths
_PKG_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR    = os.path.join(_PKG_ROOT, "data")
_CACHE_PATH  = os.path.join(_DATA_DIR, "live_latest.nc")
_STATUS_PATH = os.path.join(_DATA_DIR, "live_status.json")

# Coverage area: Arabian Sea + Bay of Bengal + Indian subcontinent
# Slightly wider than Biparjoy box so it catches any Arabian Sea/BoB storm
AREA_N, AREA_S = 30.0,  5.0
AREA_W, AREA_E = 55.0, 100.0

# -----------------------------------------------------------------------------
# Source 1 -- Open-Meteo (free, no auth, JSON API)
# -----------------------------------------------------------------------------

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

def _fetch_openmeteo_grid(
    lat_range: Tuple[float, float] = (AREA_S, AREA_N),
    lon_range: Tuple[float, float] = (AREA_W, AREA_E),
    resolution: float = 1.0,           # degrees
    days: int = 3,
) -> Optional[xr.Dataset]:
    """
    Fetch a coarse grid of forecast data from Open-Meteo for the Indian Ocean
    region by querying a regular lat/lon grid of points.

    Open-Meteo is a free, no-auth API that delivers GFS/ERA5-driven hourly data.
    It returns JSON per point; we assemble a lightweight xr.Dataset.

    NOTE: We intentionally keep the grid coarse (1° resolution, ~111 km spacing)
    to avoid hammering the API. The downscaler then takes this to 5 km.
    """
    lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)

    # Sample a sub-grid of ≤ 25 points (5x5) to stay within rate limits
    lat_idx = np.linspace(0, len(lats) - 1, min(5, len(lats)), dtype=int)
    lon_idx = np.linspace(0, len(lons) - 1, min(5, len(lons)), dtype=int)
    sample_lats = lats[lat_idx]
    sample_lons = lons[lon_idx]

    print(f"  [open-meteo] Fetching {len(sample_lats)}x{len(sample_lons)} "
          f"grid points over Indian Ocean ...")

    records: Dict[str, list] = {
        "time": [], "lat": [], "lon": [],
        "mslp": [], "u10": [], "v10": [], "precipitation": []
    }

    for lat in sample_lats:
        for lon in sample_lons:
            params = {
                "latitude":     lat,
                "longitude":    lon,
                "hourly":       ["surface_pressure", "wind_speed_10m",
                                 "wind_direction_10m", "precipitation"],
                "forecast_days": days,
                "timezone":     "UTC",
                "wind_speed_unit": "kmh",
            }
            try:
                r = requests.get(OPENMETEO_URL, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                print(f"    [warn] ({lat:.1f}°N, {lon:.1f}°E): {exc}")
                continue

            h   = data.get("hourly", {})
            times      = pd.to_datetime(h.get("time", []))
            pressures  = h.get("surface_pressure", [])
            speeds     = h.get("wind_speed_10m", [])
            directions = h.get("wind_direction_10m", [])
            precip     = h.get("precipitation", [])

            for i, t in enumerate(times):
                if i >= len(pressures):
                    break
                spd = speeds[i] if i < len(speeds) and speeds[i] is not None else 0.0
                dr  = directions[i] if i < len(directions) and directions[i] is not None else 0.0
                dr_rad = np.radians(dr)
                records["time"].append(t)
                records["lat"].append(lat)
                records["lon"].append(lon)
                records["mslp"].append(pressures[i] if pressures[i] is not None else 1013.0)
                records["u10"].append(-spd * np.sin(dr_rad))    # meteorological convention
                records["v10"].append(-spd * np.cos(dr_rad))
                records["precipitation"].append(precip[i] if i < len(precip) and precip[i] is not None else 0.0)

    if not records["time"]:
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    # Pivot to (time, lat, lon) grid
    times_unique = sorted(df["time"].unique())
    lats_unique  = sorted(df["lat"].unique())
    lons_unique  = sorted(df["lon"].unique())

    shape  = (len(times_unique), len(lats_unique), len(lons_unique))
    t_idx  = {t: i for i, t in enumerate(times_unique)}
    la_idx = {la: i for i, la in enumerate(lats_unique)}
    lo_idx = {lo: i for i, lo in enumerate(lons_unique)}

    grids  = {v: np.full(shape, np.nan, dtype=np.float32)
              for v in ("mslp", "u10", "v10", "precipitation")}

    for _, row in df.iterrows():
        ti = t_idx[row["time"]]
        li = la_idx[row["lat"]]
        oi = lo_idx[row["lon"]]
        for v in grids:
            grids[v][ti, li, oi] = row[v]

    # Fill NaNs with column mean (missing points)
    for v in grids:
        col_mean = np.nanmean(grids[v])
        grids[v] = np.where(np.isnan(grids[v]), col_mean, grids[v])

    ds = xr.Dataset(
        {
            "mslp":          (["time", "lat", "lon"], grids["mslp"],
                              {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
            "u10":           (["time", "lat", "lon"], grids["u10"],
                              {"units": "km/h", "long_name": "10m U-wind component"}),
            "v10":           (["time", "lat", "lon"], grids["v10"],
                              {"units": "km/h", "long_name": "10m V-wind component"}),
            "precipitation": (["time", "lat", "lon"], grids["precipitation"],
                              {"units": "mm/h", "long_name": "Precipitation rate"}),
        },
        coords={
            "time": np.array(times_unique, dtype="datetime64[ns]"),
            "lat":  np.array(lats_unique, dtype=np.float32),
            "lon":  np.array(lons_unique, dtype=np.float32),
        },
        attrs={
            "title":      "Meteora: Live GFS forecast via Open-Meteo API",
            "source":     "Open-Meteo (GFS/ERA5 blend), free tier",
            "provenance": "LIVE_OPENMETEO",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "area":       f"N{AREA_N} S{AREA_S} W{AREA_W} E{AREA_E}",
        },
    )
    print(f"  [open-meteo] [OK]  {len(times_unique)} timesteps, "
          f"{len(lats_unique)}×{len(lons_unique)} grid")
    return ds


# -----------------------------------------------------------------------------
# Source 2 -- GFS via NCEP/NCMRWF OpenDAP (THREDDS catalog)
# -----------------------------------------------------------------------------

GFS_THREDDS_BASE = (
    "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date}/gfs_0p25_00z"
)


def _fetch_gfs_opendap(
    lat_range: Tuple[float, float] = (AREA_S, AREA_N),
    lon_range: Tuple[float, float] = (AREA_W, AREA_E),
) -> Optional[xr.Dataset]:
    """
    Fetch the latest GFS 0.25° operational forecast from NCEP NOMADS via
    OpenDAP.  Returns up to 24 timesteps (0–72 h in 3-h steps) over the
    Indian Ocean bounding box.

    No authentication required.  Data is the actual NWP operational output
    used by IMD and NDRF for their operational guidance.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    url   = GFS_THREDDS_BASE.format(date=today)
    print(f"  [gfs-opendap] Trying NCEP NOMADS: {url}")

    try:
        ds_raw = xr.open_dataset(url, engine="pydap")
    except Exception as exc:
        # Try yesterday if today's run isn't posted yet
        yesterday = (datetime.now(timezone.utc) - pd.Timedelta("1D")).strftime("%Y%m%d")
        url_yd    = GFS_THREDDS_BASE.format(date=yesterday)
        print(f"  [gfs-opendap] Today's run not ready; trying {yesterday}: {url_yd}")
        try:
            ds_raw = xr.open_dataset(url_yd, engine="pydap")
        except Exception as exc2:
            print(f"  [gfs-opendap] [X]  Both failed: {exc2}")
            return None

    # Subset to Indian Ocean area and first 24 lead times
    try:
        ds_sub = ds_raw.sel(
            lat=slice(lat_range[0], lat_range[1]),
            lon=slice(lon_range[0], lon_range[1]),
        ).isel(time=slice(0, 24))

        rename_map = {}
        if "prmslmsl" in ds_sub:   rename_map["prmslmsl"] = "mslp_pa"
        if "ugrd10m"  in ds_sub:   rename_map["ugrd10m"]  = "u10_ms"
        if "vgrd10m"  in ds_sub:   rename_map["vgrd10m"]  = "v10_ms"
        if "apcpsfc"  in ds_sub:   rename_map["apcpsfc"]  = "precip_kgm2"
        if rename_map:
            ds_sub = ds_sub.rename(rename_map)

        # Unit conversions
        out_vars = {}
        if "mslp_pa" in ds_sub:
            out_vars["mslp"] = ds_sub["mslp_pa"] / 100.0  # Pa -> hPa
        if "u10_ms" in ds_sub:
            out_vars["u10"] = ds_sub["u10_ms"] * 3.6       # m/s -> km/h
        if "v10_ms" in ds_sub:
            out_vars["v10"] = ds_sub["v10_ms"] * 3.6
        if "precip_kgm2" in ds_sub:
            out_vars["precipitation"] = ds_sub["precip_kgm2"]  # kg/m² ≈ mm

        if not out_vars:
            print(f"  [gfs-opendap] [X]  No expected variables found. "
                  f"Got: {list(ds_sub.data_vars)}")
            return None

        ds_out = xr.Dataset(out_vars)
        ds_out.attrs.update({
            "title":      "Meteora: Live GFS 0.25° forecast (NCEP NOMADS)",
            "source":     "NCEP GFS via NOMADS OpenDAP",
            "provenance": "LIVE_GFS_OPENDAP",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
        ds_raw.close()
        print(f"  [gfs-opendap] [OK]  shape={dict(ds_out.sizes)}")
        return ds_out

    except Exception as exc:
        print(f"  [gfs-opendap] [X]  Subsetting/rename failed: {exc}")
        try:
            ds_raw.close()
        except Exception:
            pass
        return None


# -----------------------------------------------------------------------------
# Source 3 -- ERA5 CDS (if configured)
# -----------------------------------------------------------------------------

def _fetch_era5_latest() -> Optional[xr.Dataset]:
    """
    Fetch the last 24 hours of ERA5 reanalysis from Copernicus CDS.
    Requires ~/.cdsapirc with valid credentials.
    ERA5 has a 5-day lag, so 'latest' means 5 days ago.
    """
    try:
        import cdsapi
    except ImportError:
        return None

    cdsapirc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsapirc):
        return None

    print("  [era5-cds] ~/.cdsapirc found; fetching last available ERA5 ...")
    lag_date = (datetime.now(timezone.utc) - pd.Timedelta("5D")).strftime("%Y-%m-%d")
    tmp_raw  = os.path.join(_DATA_DIR, "live_era5_raw.nc")

    try:
        client = cdsapi.Client(quiet=True)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable":     ["mean_sea_level_pressure",
                                 "10m_u_component_of_wind",
                                 "10m_v_component_of_wind",
                                 "total_precipitation"],
                "year":   [lag_date[:4]],
                "month":  [lag_date[5:7]],
                "day":    [lag_date[8:10]],
                "time":   ["00:00", "06:00", "12:00", "18:00"],
                "area":   [AREA_N, AREA_W, AREA_S, AREA_E],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            tmp_raw,
        )
        from core.data_loader import normalise_era5
        raw = xr.open_dataset(tmp_raw)
        ds  = normalise_era5(raw)
        raw.close()
        ds.attrs["provenance"] = "LIVE_ERA5_CDS"
        ds.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
        print(f"  [era5-cds] [OK]  {dict(ds.sizes)}")
        return ds
    except Exception as exc:
        print(f"  [era5-cds] [X]  {exc}")
        return None


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

class LiveIngestionPipeline:
    """
    Orchestrates live weather data ingestion with automatic source fallback.

    Fetch order:
        1. Open-Meteo (no auth, fast JSON)
        2. GFS NOMADS OpenDAP (no auth, real NWP)
        3. ERA5 CDS (requires credentials)
        4. Static biparjoy_real.nc

    Results are cached to data/live_latest.nc to avoid repeated API calls.
    Cache is considered fresh for CACHE_TTL_SECONDS (default: 3 hours).
    """
    CACHE_TTL_SECONDS: int = 3 * 3600   # 3 hours

    def __init__(self):
        os.makedirs(_DATA_DIR, exist_ok=True)

    def _cache_is_fresh(self) -> bool:
        if not os.path.exists(_STATUS_PATH):
            return False
        try:
            with open(_STATUS_PATH) as f:
                status = json.load(f)
            fetched = datetime.fromisoformat(status["fetched_at"])
            age     = (datetime.now(timezone.utc) - fetched).total_seconds()
            return age < self.CACHE_TTL_SECONDS
        except Exception:
            return False

    def _save_cache(self, ds: xr.Dataset) -> None:
        ds.to_netcdf(_CACHE_PATH)
        status = {
            "fetched_at":  datetime.now(timezone.utc).isoformat(),
            "source":      ds.attrs.get("provenance", "UNKNOWN"),
            "timesteps":   int(ds.sizes.get("time", 0)),
            "lat_range":   [float(ds.lat.min()), float(ds.lat.max())],
            "lon_range":   [float(ds.lon.min()), float(ds.lon.max())],
            "variables":   list(ds.data_vars),
        }
        with open(_STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2)

    def get_status(self) -> Dict:
        """Return the status of the most recent ingestion."""
        if not os.path.exists(_STATUS_PATH):
            return {"source": "none", "fetched_at": None, "fresh": False}
        try:
            with open(_STATUS_PATH) as f:
                status = json.load(f)
            status["fresh"] = self._cache_is_fresh()
            return status
        except Exception:
            return {"source": "error", "fetched_at": None, "fresh": False}

    def get_latest(self) -> xr.Dataset:
        """
        Return the latest cached dataset without hitting the network.
        Falls back to biparjoy_real.nc if no cache exists.
        """
        if os.path.exists(_CACHE_PATH):
            try:
                return xr.open_dataset(_CACHE_PATH)
            except Exception:
                pass
        from core.data_loader import load_biparjoy_data
        return load_biparjoy_data()

    def fetch(self, force: bool = False) -> xr.Dataset:
        """
        Fetch fresh live data. Uses cache if fresh and force=False.

        Parameters
        ----------
        force : bypass cache and always fetch from the network

        Returns
        -------
        xr.Dataset in Meteora canonical schema
        """
        if not force and self._cache_is_fresh():
            print("[ingest] Cache is fresh; skipping network fetch.")
            return self.get_latest()

        print("[ingest] Starting live data fetch ...")
        t0 = time.time()
        ds = None

        # -- Try Source 1: Open-Meteo ------------------------------------
        print("[ingest] Source 1: Open-Meteo ...")
        try:
            ds = _fetch_openmeteo_grid()
        except Exception as exc:
            print(f"  Open-Meteo failed: {exc}")

        # -- Try Source 2: GFS OpenDAP -----------------------------------
        if ds is None:
            print("[ingest] Source 2: GFS NOMADS OpenDAP ...")
            try:
                ds = _fetch_gfs_opendap()
            except Exception as exc:
                print(f"  GFS OpenDAP failed: {exc}")

        # -- Try Source 3: ERA5 CDS --------------------------------------
        if ds is None:
            print("[ingest] Source 3: ERA5 CDS ...")
            try:
                ds = _fetch_era5_latest()
            except Exception as exc:
                print(f"  ERA5 CDS failed: {exc}")

        # -- Fallback: static file ---------------------------------------
        if ds is None:
            print("[ingest] [!]  All live sources failed; using static biparjoy_real.nc")
            from core.data_loader import load_biparjoy_data
            ds = load_biparjoy_data()

        elapsed = time.time() - t0
        src     = ds.attrs.get("provenance", "UNKNOWN")
        print(f"[ingest] [OK]  Ingestion complete in {elapsed:.1f}s  source={src}")

        # Cache the result
        try:
            self._save_cache(ds)
            print(f"[ingest] Cached -> {_CACHE_PATH}")
        except Exception as exc:
            print(f"[ingest] Cache write failed (non-fatal): {exc}")

        return ds


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def run_live_ingest(force: bool = True) -> Dict:
    """Convenience function for CLI and API use."""
    pipeline = LiveIngestionPipeline()
    ds       = pipeline.fetch(force=force)
    status   = pipeline.get_status()
    ds.close()
    return status


if __name__ == "__main__":
    import pprint
    result = run_live_ingest(force=True)
    pprint.pprint(result)
