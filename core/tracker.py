# core/tracker.py
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import xarray as xr
import pandas as pd


class AnomalyTracker:
    """
    Stage 1: Multi-timestep weather grids ko scan karta hai,
    extreme pressure drops aur cyclonic vorticity detect karta hai,
    aur dynamic 4D bounding boxes construct karta hai.
    """
    def __init__(self, dataset_path: Optional[str] = None):
        from .data_loader import load_biparjoy_data, REAL_CANONICAL, SYNTHETIC_CANONICAL
        if dataset_path is not None and os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
            self.dataset_path = dataset_path
            self.ds = xr.open_dataset(dataset_path)
        else:
            self.ds = load_biparjoy_data()
            self.dataset_path = REAL_CANONICAL if os.path.exists(REAL_CANONICAL) else SYNTHETIC_CANONICAL

        self.times = self.ds.time.values
        
    def detect_storm_center(self, time_idx: int) -> Dict:
        """
        Single timestep par minimum pressure centroid aur maximum wind intensity extract karta hai.
        """
        slice_t = self.ds.isel(time=time_idx)
        
        # 1. Centroid spot karna: Minimum MSLP location (Cyclone Eye)
        var_name = "mslp" if "mslp" in slice_t.data_vars else "msl"
        mslp_arr = slice_t[var_name].values
        min_idx = np.unravel_index(np.argmin(mslp_arr), mslp_arr.shape)
        
        center_lat = float(slice_t["lat"].values[min_idx[0]])
        center_lon = float(slice_t["lon"].values[min_idx[1]])
        min_mslp = float(mslp_arr[min_idx])
        
        # 2. Maximum Wind Speed calculate karna: sqrt(u^2 + v^2)
        if "u10" in slice_t.data_vars and "v10" in slice_t.data_vars:
            wind_mag = np.sqrt(slice_t["u10"].values**2 + slice_t["v10"].values**2)
            max_wind = float(np.max(wind_mag))
        elif "wind_speed" in slice_t.data_vars:
            max_wind = float(np.max(slice_t["wind_speed"].values))
        else:
            delta_p = max(0.0, 1012.0 - min_mslp)
            max_wind = float(14.2 * np.sqrt(delta_p) * 1.852)

        if "precipitation" in slice_t.data_vars:
            max_precip = float(np.max(slice_t["precipitation"].values))
        else:
            max_precip = 0.0
        
        # 3. Dynamic Bounding Box Dimensions (+/- 1.8 degree window approx 200 km radius)
        lat_buffer = 1.8
        lon_buffer = 1.8
        
        bbox = {
            "lat_min": round(max(float(self.ds.lat.min()), center_lat - lat_buffer), 3),
            "lat_max": round(min(float(self.ds.lat.max()), center_lat + lat_buffer), 3),
            "lon_min": round(max(float(self.ds.lon.min()), center_lon - lon_buffer), 3),
            "lon_max": round(min(float(self.ds.lon.max()), center_lon + lon_buffer), 3),
        }
        
        # 4. Severity Category calculate karna based on central pressure
        if min_mslp < 970.0:
            category = "Extremely Severe Cyclonic Storm (ESCS)"
            alert_level = "CRITICAL"
        elif min_mslp < 985.0:
            category = "Very Severe Cyclonic Storm (VSCS)"
            alert_level = "HIGH"
        elif min_mslp < 995.0:
            category = "Severe Cyclonic Storm (SCS)"
            alert_level = "MODERATE"
        else:
            category = "Deep Depression / Tropical Storm"
            alert_level = "ADVISORY"
            
        timestamp_str = pd.to_datetime(str(slice_t.time.values)).strftime("%Y-%m-%d %H:%M UTC")
        
        return {
            "time_idx": time_idx,
            "timestamp": timestamp_str,
            "centroid": (center_lat, center_lon),
            "min_mslp_hpa": round(min_mslp, 2),
            "max_wind_kmh": round(max_wind, 2),
            "max_precip_mmh": round(max_precip, 2),
            "category": category,
            "alert_level": alert_level,
            "bbox": bbox
        }

    def get_full_trajectory(self) -> List[Dict]:
        """
        Poore forecast lead time (T=0 se T=+72h) ka complete tracking path return karta hai.
        """
        trajectory = []
        for t in range(len(self.times)):
            trajectory.append(self.detect_storm_center(t))
        return trajectory

    def crop_anomaly_subgrid(self, time_idx: int) -> xr.Dataset:
        """
        Stage 2 Downscaler ke liye sirf bounded anomaly area ko crop karta hai.
        Isse pure continent par diffusion chalane ka VRAM overhead khatam ho jata hai.
        """
        detection = self.detect_storm_center(time_idx)
        b = detection["bbox"]
        
        # Bounding box slice extraction
        lat_slice = slice(b["lat_min"], b["lat_max"]) if float(self.ds.lat[1]) > float(self.ds.lat[0]) else slice(b["lat_max"], b["lat_min"])
        lon_slice = slice(b["lon_min"], b["lon_max"]) if float(self.ds.lon[1]) > float(self.ds.lon[0]) else slice(b["lon_max"], b["lon_min"])

        cropped_ds = self.ds.isel(time=time_idx).sel(
            lat=lat_slice,
            lon=lon_slice
        )
        return cropped_ds


# --- Backwards compatibility wrappers for earlier pipeline modules ---
def create_sample_biparjoy_netcdf(filepath: str = "data/biparjoy_sample.nc") -> None:
    from .data_loader import generate_biparjoy_data
    generate_biparjoy_data(filepath)


def extract_anomaly_bounding_box(
    ds: xr.Dataset,
    time_idx: int = 0,
    threshold_hpa: float = 990.0
) -> Optional[Dict[str, Any]]:
    var_name = "mslp" if "mslp" in ds.data_vars else "msl"
    slice_data = ds.isel(time=time_idx)
    min_point = slice_data[var_name].where(slice_data[var_name] < threshold_hpa, drop=True)
    
    if min_point.size == 0:
        return None

    min_lat = float(min_point.lat.min())
    max_lat = float(min_point.lat.max())
    min_lon = float(min_point.lon.min())
    max_lon = float(min_point.lon.max())

    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    return {
        "time": str(slice_data.time.values),
        "centroid": (round(center_lat, 4), round(center_lon, 4)),
        "bounds": {
            "lat_min": round(center_lat - 1.5, 4),
            "lat_max": round(center_lat + 1.5, 4),
            "lon_min": round(center_lon - 1.5, 4),
            "lon_max": round(center_lon + 1.5, 4)
        }
    }


class CycloneTracker(AnomalyTracker):
    """
    Adapter extending AnomalyTracker with earlier API methods
    (get_slice, extract_centroid, extract_bounding_box, process_timestep, track_all).
    """
    def __init__(self, nc_path: str = "data/biparjoy_sample.nc", var_name: Optional[str] = None, ambient_pressure: float = 1012.0):
        super().__init__(dataset_path=nc_path)
        self.nc_path = self.dataset_path
        self.ambient_pressure = ambient_pressure
        self.var_name = var_name or ("mslp" if "mslp" in self.ds.data_vars else "msl")
        self.lats = self.ds["lat"].values
        self.lons = self.ds["lon"].values

    def get_slice(self, time_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        slice_da = self.ds[self.var_name].isel(time=time_index)
        pressure_2d = slice_da.values.astype(np.float64)
        ts_str = str(np.datetime_as_string(self.times[time_index], unit="s"))
        return pressure_2d, self.lats, self.lons, ts_str

    def extract_centroid(self, pressure_2d: np.ndarray) -> Tuple[float, float, float]:
        min_idx = np.unravel_index(np.argmin(pressure_2d), pressure_2d.shape)
        min_lat = float(self.lats[min_idx[0]])
        min_lon = float(self.lons[min_idx[1]])
        min_p = float(pressure_2d[min_idx])
        return round(min_lat, 4), round(min_lon, 4), round(min_p, 2)

    def extract_bounding_box(self, centroid_lat: float, centroid_lon: float, radius_deg: float = 1.8) -> List[float]:
        min_lon = max(float(self.lons.min()), centroid_lon - radius_deg)
        max_lon = min(float(self.lons.max()), centroid_lon + radius_deg)
        min_lat = max(float(self.lats.min()), centroid_lat - radius_deg)
        max_lat = min(float(self.lats.max()), centroid_lat + radius_deg)
        return [round(min_lon, 3), round(min_lat, 3), round(max_lon, 3), round(max_lat, 3)]

    def estimate_intensity(self, min_pressure: float) -> Dict[str, Any]:
        delta_p = max(0.0, self.ambient_pressure - min_pressure)
        vmax_knots = 14.2 * np.sqrt(delta_p) if delta_p > 0 else 0.0
        vmax_kmh = vmax_knots * 1.852

        if min_pressure < 970.0:
            category = "Extremely Severe Cyclonic Storm (ESCS)"
            alert_level = "CRITICAL"
        elif min_pressure < 985.0:
            category = "Very Severe Cyclonic Storm (VSCS)"
            alert_level = "HIGH"
        elif min_pressure < 995.0:
            category = "Severe Cyclonic Storm (SCS)"
            alert_level = "MODERATE"
        else:
            category = "Deep Depression / Tropical Storm"
            alert_level = "ADVISORY"

        return {
            "category": category,
            "alert_level": alert_level,
            "vmax_knots": round(float(vmax_knots), 1),
            "vmax_kmh": round(float(vmax_kmh), 1),
            "pressure_deficit_hpa": round(float(delta_p), 2),
        }

    def process_timestep(self, time_index: int) -> Dict[str, Any]:
        det = self.detect_storm_center(time_index)
        bbox_list = [det["bbox"]["lon_min"], det["bbox"]["lat_min"], det["bbox"]["lon_max"], det["bbox"]["lat_max"]]
        return {
            "time_index": time_index,
            "timestamp": det["timestamp"],
            "centroid": {"lat": det["centroid"][0], "lon": det["centroid"][1]},
            "min_pressure_hpa": det["min_mslp_hpa"],
            "bounding_box": bbox_list,
            "intensity": {
                "category": det["category"],
                "alert_level": det["alert_level"],
                "vmax_kmh": det["max_wind_kmh"],
                "vmax_knots": round(det["max_wind_kmh"] / 1.852, 1),
                "pressure_deficit_hpa": round(max(0.0, 1012.0 - det["min_mslp_hpa"]), 2),
            },
        }

    def track_all(self) -> List[Dict[str, Any]]:
        return [self.process_timestep(i) for i in range(len(self.times))]
