# core/data_loader.py
"""
Data ingestion for the Meteora pipeline.

Two sources, in priority order:

1. REAL -- ERA5 reanalysis (Copernicus CDS) for Cyclone Biparjoy, June 2023.
   Downloaded by scripts/fetch_era5.py, then normalised here into the canonical
   schema that tracker.py / downscaler.py consume.  This is observational data.

2. SYNTHETIC -- an analytic modified-Rankine vortex, kept ONLY as an offline
   fallback so the repo runs without a network or a CDS account.  Every
   consumer must label output derived from it as SIMULATED, never as
   an observation of the real cyclone.

Canonical schema produced by both paths:

    time            datetime64[ns]
    lat             ascending,  degrees_north
    lon             ascending,  degrees_east  (-180..180)
    mslp            mean sea level pressure          hPa
    u10, v10        10 m wind components             km/h
    precipitation   rainfall rate                    mm/h
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# Paths, relative to the meteora-prototype/ package root.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
RAW_ERA5 = os.path.join(DATA_DIR, "era5_biparjoy_2023_raw.nc")
REAL_CANONICAL = os.path.join(DATA_DIR, "biparjoy_real.nc")
SYNTHETIC_CANONICAL = os.path.join(DATA_DIR, "biparjoy_sample.nc")

# Provenance flag, read by the UI so it can label the source honestly.
SOURCE_REAL = "ERA5_REANALYSIS"
SOURCE_SYNTHETIC = "SYNTHETIC_SIMULATION"


# --------------------------------------------------------------------------- #
# Real data: ERA5 -> canonical schema
# --------------------------------------------------------------------------- #

def normalise_era5(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a raw CDS ERA5 single-levels file into the Meteora canonical schema.

    Handles the four things that silently break downstream code if ignored:
      * coordinate naming   (valid_time/latitude/longitude -> time/lat/lon)
      * the `expver` pseudo-dimension CDS adds for preliminary vs final data
      * longitude convention (0..360 -> -180..180) and descending latitude
      * unit conversion     (Pa->hPa, m/s->km/h, m->mm)

    Raises ValueError if a required variable is missing, rather than letting a
    KeyError surface three modules later.
    """
    ds = ds.copy()

    # 1. Coordinate renaming (new CDS NetCDF uses valid_time/latitude/longitude).
    rename = {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.coords or k in ds.dims})

    # 2. expver: CDS serves preliminary (5) and final (1) data joined on this
    #    axis. Exactly one carries values at any given time; take whichever is
    #    populated instead of averaging in a NaN and poisoning the grid.
    if "expver" in ds.dims:
        if ds.sizes["expver"] == 1:
            ds = ds.isel(expver=0, drop=True)
        else:
            a, b = ds.expver.values[:2]
            ds = ds.sel(expver=a).combine_first(ds.sel(expver=b)).drop_vars(
                "expver", errors="ignore"
            )

    # 3. Squeeze any other singleton dimensions (surface, number, ...).
    singleton = [d for d in ds.dims if ds.sizes[d] == 1 and d not in ("time", "lat", "lon")]
    if singleton:
        ds = ds.squeeze(singleton, drop=True)

    # 4. Required variables.
    missing = [v for v in ("msl", "u10", "v10") if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"ERA5 file is missing required variable(s) {missing}. "
            f"Found: {sorted(ds.data_vars)}. "
            "Re-run scripts/fetch_era5.py with the full variable list."
        )

    # 5. Longitude convention: ERA5 ships 0..360. Shift to -180..180 and sort,
    #    otherwise a bounding box near 0 degrees silently wraps the globe.
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180.0) % 360.0) - 180.0)).sortby("lon")

    # 6. Latitude must be ascending for the slice() logic in tracker.py.
    if "lat" in ds.coords and float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sortby("lat")

    # 7. Units.
    ds["mslp"] = ds["msl"] / 100.0                       # Pa    -> hPa
    ds["u10"] = ds["u10"] * 3.6                          # m/s   -> km/h
    ds["v10"] = ds["v10"] * 3.6                          # m/s   -> km/h

    if "tp" in ds.data_vars:
        ds["precipitation"] = _deaccumulate_precipitation(ds["tp"], ds["time"])

    ds = ds.drop_vars([v for v in ("msl", "tp") if v in ds.data_vars])

    ds.attrs.update(
        {
            "title": "Meteora: Cyclone Biparjoy (Arabian Sea) - ERA5 reanalysis",
            "source": "ECMWF ERA5 via Copernicus Climate Data Store",
            "provenance": SOURCE_REAL,
            "note": (
                "Real reanalysis at 0.25 deg (~31 km). Wind converted to km/h, "
                "pressure to hPa, precipitation de-accumulated to mm/h."
            ),
        }
    )
    for name, units in (
        ("mslp", "hPa"),
        ("u10", "km/h"),
        ("v10", "km/h"),
        ("precipitation", "mm/h"),
    ):
        if name in ds:
            ds[name].attrs["units"] = units

    return ds


def _deaccumulate_precipitation(tp: xr.DataArray, time: xr.DataArray) -> xr.DataArray:
    """
    ERA5 `tp` is an accumulation since a reset point (in practice, 00 UTC of
    each day), not an instantaneous rate. Convert it to mm/h.

    Strategy: take the time increment of the accumulated field. A negative
    increment means the accumulator reset, so the value itself is the amount
    accumulated since the reset. Divide by the step length to get a rate.

    NOTE: the reset timing is an ERA5 convention, not something this file
    states. The sanity check below warns if the assumption produces physically
    impossible (negative) rates, so the failure is visible rather than silent.
    """
    tp_mm = tp * 1000.0  # m -> mm
    steps = np.diff(time.values).astype("timedelta64[m]").astype(float) / 60.0
    steps = np.concatenate([[steps[0]], steps])  # first step reuses the second

    increment = np.diff(tp_mm.values, axis=0, prepend=tp_mm.values[:1])
    reset = increment < 0.0
    increment = np.where(reset, tp_mm.values, increment)

    rate = increment / steps.reshape((-1,) + (1,) * (tp_mm.ndim - 1))

    if np.any(rate < -1e-6):
        warnings.warn(
            "De-accumulated precipitation contains negative rates; the ERA5 "
            "accumulation-reset assumption may not hold for this request. "
            "Treat `precipitation` as approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return xr.DataArray(
        np.maximum(rate, 0.0).astype(np.float32),
        coords=tp_mm.coords,
        dims=tp_mm.dims,
        name="precipitation",
    )


def open_raw_era5(raw_path: str = RAW_ERA5) -> xr.Dataset:
    """
    Open raw ERA5 download. If CDS packaged the result as a zip archive
    (because multiple stepTypes like instant and accum were requested),
    extract and merge the component NetCDF files.
    """
    import zipfile

    if zipfile.is_zipfile(raw_path):
        extract_dir = os.path.join(os.path.dirname(raw_path), "era5_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(raw_path) as z:
            z.extractall(extract_dir)
        nc_files = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if f.endswith(".nc")
        ]
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found in archive {raw_path}")
        datasets = [xr.open_dataset(p) for p in nc_files]
        if len(datasets) == 1:
            return datasets[0]
        return xr.merge(datasets, compat="override")
    return xr.open_dataset(raw_path)


def build_canonical_dataset(raw_path: str = RAW_ERA5, out_path: str = REAL_CANONICAL) -> str:
    """Normalise a raw ERA5 download and write the canonical NetCDF for the pipeline."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw ERA5 file not found: {raw_path}\n"
            "Run `python scripts/fetch_era5.py` first."
        )

    raw = open_raw_era5(raw_path)
    canonical = normalise_era5(raw)
    raw.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canonical.to_netcdf(out_path)
    canonical.close()

    print(f"[OK] canonical dataset -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Synthetic fallback (offline only -- NOT observational data)
# --------------------------------------------------------------------------- #

def generate_biparjoy_data(output_path: str = SYNTHETIC_CANONICAL) -> str:
    """
    Generate a SYNTHETIC Biparjoy-shaped dataset from an analytic modified
    Rankine vortex, for offline development and tests.

    This is a plausible-looking simulation, NOT a reanalysis or an observation.
    Nothing derived from it may be presented as a real forecast or a real storm.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    rng = np.random.default_rng(seed=20230615)  # fixed seed -> reproducible

    times = pd.date_range("2023-06-12 00:00", periods=7, freq="12h")
    lats = np.linspace(16.0, 24.5, 85)
    lons = np.linspace(65.0, 72.5, 75)

    # Approximate published Biparjoy track: Arabian Sea -> Jakhau/Mandvi landfall.
    storm_trajectory = [
        {"lat": 18.5, "lon": 67.2, "central_mslp": 965.0, "max_wind": 145.0},
        {"lat": 19.4, "lon": 67.4, "central_mslp": 962.0, "max_wind": 150.0},
        {"lat": 20.6, "lon": 67.8, "central_mslp": 968.0, "max_wind": 140.0},
        {"lat": 21.7, "lon": 68.3, "central_mslp": 972.0, "max_wind": 135.0},
        {"lat": 22.5, "lon": 68.8, "central_mslp": 978.0, "max_wind": 125.0},
        {"lat": 23.2, "lon": 69.4, "central_mslp": 985.0, "max_wind": 110.0},
        {"lat": 23.8, "lon": 70.2, "central_mslp": 994.0, "max_wind": 75.0},
    ]

    T, Y, X = len(times), len(lats), len(lons)
    mslp_grid = np.zeros((T, Y, X), dtype=np.float32)
    wind_u = np.zeros((T, Y, X), dtype=np.float32)
    wind_v = np.zeros((T, Y, X), dtype=np.float32)
    precip_grid = np.zeros((T, Y, X), dtype=np.float32)
    moisture_grid = np.zeros((T, Y, X), dtype=np.float32)

    LON, LAT = np.meshgrid(lons, lats)

    for t_idx, state in enumerate(storm_trajectory):
        c_lat, c_lon = state["lat"], state["lon"]
        p_drop = 1010.0 - state["central_mslp"]
        v_max = state["max_wind"]

        dx = (LON - c_lon) * 111.0 * np.cos(np.radians(c_lat))
        dy = (LAT - c_lat) * 111.0
        dist = np.sqrt(dx**2 + dy**2)
        r_max = 45.0

        mslp_grid[t_idx] = 1012.0 - p_drop * np.exp(-((dist / 140.0) ** 1.3))

        with np.errstate(divide="ignore", invalid="ignore"):
            tangential_wind = np.where(
                dist <= r_max,
                v_max * (dist / r_max),
                v_max * ((r_max / dist) ** 0.6),
            )
            tangential_wind[dist == 0] = 0.0

        angle = np.arctan2(dy, dx)
        wind_u[t_idx] = -tangential_wind * np.sin(angle)
        wind_v[t_idx] = tangential_wind * np.cos(angle)

        precip_grid[t_idx] = np.maximum(
            0.0, (v_max / 1.6) * (dist / r_max) * np.exp(-dist / 80.0)
        )
        moisture_grid[t_idx] = 18.0 * np.exp(-dist / 250.0) + rng.normal(0, 0.4, (Y, X))

    ds = xr.Dataset(
        data_vars={
            "mslp": (("time", "lat", "lon"), mslp_grid, {"units": "hPa"}),
            "u10": (("time", "lat", "lon"), wind_u, {"units": "km/h"}),
            "v10": (("time", "lat", "lon"), wind_v, {"units": "km/h"}),
            "precipitation": (("time", "lat", "lon"), precip_grid, {"units": "mm/h"}),
            "humidity": (("time", "lat", "lon"), moisture_grid, {"units": "g/kg"}),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "title": "Meteora OFFLINE FALLBACK: synthetic Biparjoy-shaped vortex",
            "source": "Analytic modified Rankine vortex (not observational)",
            "provenance": SOURCE_SYNTHETIC,
            "warning": (
                "SIMULATED DATA. Not a reanalysis or observation. For offline "
                "development only -- must not be presented as real weather."
            ),
        },
    )

    ds.to_netcdf(output_path)
    print(f"[OK] SYNTHETIC fallback dataset written to: {output_path}")
    print(f"     time slices: {len(times)} | grid shape: {mslp_grid.shape}")
    return output_path


# --------------------------------------------------------------------------- #
# Entry point used by app.py / api.py
# --------------------------------------------------------------------------- #

def load_biparjoy_data(filepath: str | None = None, force_synthetic: bool = False) -> xr.Dataset:
    """
    Open the best available Biparjoy dataset.

    Prefers real ERA5 reanalysis; falls back to the synthetic vortex with a
    loud warning so the source can never be mistaken in a demo.
    """
    if filepath:
        return xr.open_dataset(filepath)

    if not force_synthetic and os.path.exists(REAL_CANONICAL):
        return xr.open_dataset(REAL_CANONICAL)

    if not force_synthetic and os.path.exists(RAW_ERA5):
        print("[data] real ERA5 download found; normalising...")
        return xr.open_dataset(build_canonical_dataset())

    warnings.warn(
        "No real ERA5 data found -- falling back to the SYNTHETIC vortex. "
        "Output is SIMULATED, not a real forecast. Run scripts/fetch_era5.py "
        "to use real reanalysis.",
        RuntimeWarning,
        stacklevel=2,
    )
    generate_biparjoy_data(SYNTHETIC_CANONICAL)
    return xr.open_dataset(SYNTHETIC_CANONICAL)


def dataset_provenance(ds: xr.Dataset) -> str:
    """Return the provenance tag so the UI can label real vs simulated data."""
    return str(ds.attrs.get("provenance", SOURCE_SYNTHETIC))
