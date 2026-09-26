# core/data_loader.py
import os
import numpy as np
import xarray as xr
import pandas as pd

def generate_biparjoy_data(output_path="data/biparjoy_sample.nc"):
    """
    Cyclone Biparjoy (June 2023) ke actual Arabian Sea track aur landfall 
    ke physical parameters par aadharit CF-compliant 4D NetCDF dataset generate karta hai.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # 1. Spatio-Temporal Grid Setup (Arabian Sea se Gujarat Coast)
    # Timestamps: 12 June 2023 se 15 June 2023 (12-hour intervals, 7 timesteps)
    times = pd.date_range("2023-06-12 00:00", periods=7, freq="12h")
    lats = np.linspace(16.0, 24.5, 85)   # Latitude range (12 km coarse resolution equivalent)
    lons = np.linspace(65.0, 72.5, 75)   # Longitude range

    # Biparjoy ka historical trajectory track (Lat, Lon, Central Pressure, Max Wind)
    # Mandvi / Jakhau Port landfall target (June 15)
    storm_trajectory = [
        {"lat": 18.5, "lon": 67.2, "central_mslp": 965.0, "max_wind": 145.0}, # T=0 (Severe)
        {"lat": 19.4, "lon": 67.4, "central_mslp": 962.0, "max_wind": 150.0}, # T+12h
        {"lat": 20.6, "lon": 67.8, "central_mslp": 968.0, "max_wind": 140.0}, # T+24h
        {"lat": 21.7, "lon": 68.3, "central_mslp": 972.0, "max_wind": 135.0}, # T+36h
        {"lat": 22.5, "lon": 68.8, "central_mslp": 978.0, "max_wind": 125.0}, # T+48h
        {"lat": 23.2, "lon": 69.4, "central_mslp": 985.0, "max_wind": 110.0}, # T+60h (Near Jakhau)
        {"lat": 23.8, "lon": 70.2, "central_mslp": 994.0, "max_wind": 75.0}   # T+72h (Inland Decay)
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

        # Cyclone vortex mathematics (Modified Rankine Vortex)
        dx = (LON - c_lon) * 111.0 * np.cos(np.radians(c_lat)) # distance in km
        dy = (LAT - c_lat) * 111.0                             # distance in km
        dist = np.sqrt(dx**2 + dy**2)
        r_max = 45.0  # Radius of maximum winds (45 km)

        # 1. MSLP: Hydrostatic core drop
        mslp_grid[t_idx] = 1012.0 - p_drop * np.exp(-(dist / 140.0)**1.3)

        # 2. Tangential Wind Speed profile (Rankine Vortex)
        with np.errstate(divide='ignore', invalid='ignore'):
            tangential_wind = np.where(
                dist <= r_max,
                v_max * (dist / r_max),
                v_max * ((r_max / dist)**0.6)
            )
            tangential_wind[dist == 0] = 0.0

        # Northern hemisphere cyclonic rotation (Anticlockwise: u = -V*sin, v = V*cos)
        angle = np.arctan2(dy, dx)
        wind_u[t_idx] = -tangential_wind * np.sin(angle)
        wind_v[t_idx] = tangential_wind * np.cos(angle)

        # 3. High-Intensity Rain bands near eyewall
        precip_grid[t_idx] = np.maximum(
            0.0,
            (v_max / 1.6) * (dist / r_max) * np.exp(-dist / 80.0)
        )

        # 4. Specific Humidity (Moisture convergence tracking)
        moisture_grid[t_idx] = 18.0 * np.exp(-dist / 250.0) + np.random.normal(0, 0.4, (Y, X))

    # CF-Compliant Xarray Dataset packaging
    ds = xr.Dataset(
        data_vars={
            "mslp": (("time", "lat", "lon"), mslp_grid, {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
            "u10": (("time", "lat", "lon"), wind_u, {"units": "km/h", "long_name": "10-meter U-wind component"}),
            "v10": (("time", "lat", "lon"), wind_v, {"units": "km/h", "long_name": "10-meter V-wind component"}),
            "precipitation": (("time", "lat", "lon"), precip_grid, {"units": "mm/h", "long_name": "Total Precipitation Rate"}),
            "humidity": (("time", "lat", "lon"), moisture_grid, {"units": "g/kg", "long_name": "Specific Humidity"}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons
        },
        attrs={
            "title": "Meteora Benchmark: Cyclone Biparjoy Reanalysis Slice",
            "source": "Synthesized NCMRWF/IMDAA Kinematic Profile",
            "spatial_resolution": "12 km coarse grid",
            "institution": "Meteora AI Pipeline"
        }
    )

    ds.to_netcdf(output_path)
    print(f"[OK] Biparjoy dataset successfully written to: {output_path}")
    print(f"     Time slices: {len(times)} | Grid Shape: {mslp_grid.shape}")
    return output_path

def load_biparjoy_data(filepath="data/biparjoy_sample.nc"):
    """
    Xarray out-of-core loader. Agar dataset present nahi hai toh generate karega.
    """
    if not os.path.exists(filepath):
        generate_biparjoy_data(filepath)
    return xr.open_dataset(filepath)
