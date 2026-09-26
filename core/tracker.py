"""
Stage 1: Tracker - Centroid & Bounding Box extraction for Cyclone Biparjoy.
Extracts low-pressure center (vortex core), computes sub-grid centroid,
determines dynamic bounding boxes, and generates trajectory time-series.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import xarray as xr


class CycloneTracker:
    """Stage 1: Centroid and Bounding Box Extraction for Cyclone NetCDF data."""

    def __init__(self, nc_path: str, var_name: str = "msl", ambient_pressure: float = 1012.0):
        """
        Initialize the tracker with a NetCDF dataset.

        :param nc_path: Filepath to the NetCDF file.
        :param var_name: Name of the pressure variable ('msl' in biparjoy_sample.nc).
        :param ambient_pressure: Ambient reference sea level pressure in hPa.
        """
        self.nc_path = nc_path
        self.var_name = var_name
        self.ambient_pressure = ambient_pressure
        self.ds = xr.open_dataset(nc_path)
        self.times = self.ds["time"].values
        self.lats = self.ds["lat"].values
        self.lons = self.ds["lon"].values

    def get_slice(self, time_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Get 2D pressure field slice at specified time index.

        :return: (pressure_2d, lats, lons, timestamp_str)
        """
        if time_index < 0 or time_index >= len(self.times):
            raise IndexError(f"time_index {time_index} out of range [0, {len(self.times)-1}]")

        slice_da = self.ds[self.var_name].isel(time=time_index)
        pressure_2d = slice_da.values.astype(np.float64)
        ts_str = str(np.datetime_as_string(self.times[time_index], unit="s"))
        return pressure_2d, self.lats, self.lons, ts_str

    def extract_centroid(self, pressure_2d: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract cyclone eye centroid coordinates (lat, lon) and minimum central pressure.
        Uses sub-grid weighted deficit interpolation for smooth tracking.

        :param pressure_2d: 2D array of Mean Sea Level Pressure (hPa).
        :return: (centroid_lat, centroid_lon, min_pressure_hpa)
        """
        min_idx = np.unravel_index(np.argmin(pressure_2d), pressure_2d.shape)
        min_lat_idx, min_lon_idx = min_idx
        min_p = float(pressure_2d[min_lat_idx, min_lon_idx])

        # Subgrid refinement via inverted pressure deficit weighting
        p_deficit = np.maximum(0.0, self.ambient_pressure - pressure_2d)
        
        # Local window around minimum (3x3 window)
        r_min = max(0, min_lat_idx - 1)
        r_max = min(pressure_2d.shape[0], min_lat_idx + 2)
        c_min = max(0, min_lon_idx - 1)
        c_max = min(pressure_2d.shape[1], min_lon_idx + 2)

        local_deficit = p_deficit[r_min:r_max, c_min:c_max]
        local_lats = self.lats[r_min:r_max]
        local_lons = self.lons[c_min:c_max]

        total_weight = np.sum(local_deficit)
        if total_weight > 1e-4:
            # Weighted coordinate centroid
            mesh_lon, mesh_lat = np.meshgrid(local_lons, local_lats)
            refined_lat = float(np.sum(mesh_lat * local_deficit) / total_weight)
            refined_lon = float(np.sum(mesh_lon * local_deficit) / total_weight)
        else:
            refined_lat = float(self.lats[min_lat_idx])
            refined_lon = float(self.lons[min_lon_idx])

        return round(refined_lat, 4), round(refined_lon, 4), round(min_p, 2)

    def extract_bounding_box(
        self,
        centroid_lat: float,
        centroid_lon: float,
        radius_deg: float = 3.5
    ) -> List[float]:
        """
        Extract dynamic bounding box around the cyclone core in [min_lon, min_lat, max_lon, max_lat].

        :param centroid_lat: Latitude of the center.
        :param centroid_lon: Longitude of the center.
        :param radius_deg: Influence radius in degrees (~3.5 deg = ~380 km).
        :return: [min_lon, min_lat, max_lon, max_lat] clamped to dataset boundaries.
        """
        min_lon = max(float(self.lons.min()), centroid_lon - radius_deg)
        max_lon = min(float(self.lons.max()), centroid_lon + radius_deg)
        min_lat = max(float(self.lats.min()), centroid_lat - radius_deg)
        max_lat = min(float(self.lats.max()), centroid_lat + radius_deg)

        return [round(min_lon, 3), round(min_lat, 3), round(max_lon, 3), round(max_lat, 3)]

    def estimate_intensity(self, min_pressure: float) -> Dict[str, Any]:
        """
        Estimate cyclone intensity category & estimated max wind speed (knots & km/h)
        using IMD/WMO pressure-wind relationship.
        """
        delta_p = max(0.0, self.ambient_pressure - min_pressure)
        # Empirical wind-pressure relationship for North Indian Ocean (Mishra & Gupta):
        # Vmax (knots) approx 14.2 * sqrt(delta_p in hPa)
        vmax_knots = 14.2 * np.sqrt(delta_p) if delta_p > 0 else 0.0
        vmax_kmh = vmax_knots * 1.852

        if vmax_knots >= 120 or min_pressure <= 940:
            category = "Super Cyclonic Storm"
            alert_level = "CODE_RED_EXTREME"
        elif vmax_knots >= 90 or min_pressure <= 965:
            category = "Extremely Severe Cyclonic Storm"
            alert_level = "CODE_RED"
        elif vmax_knots >= 64 or min_pressure <= 980:
            category = "Very Severe Cyclonic Storm"
            alert_level = "CODE_ORANGE"
        elif vmax_knots >= 48 or min_pressure <= 992:
            category = "Severe Cyclonic Storm"
            alert_level = "CODE_ORANGE"
        elif vmax_knots >= 34 or min_pressure <= 1000:
            category = "Cyclonic Storm"
            alert_level = "CODE_YELLOW"
        elif vmax_knots >= 17:
            category = "Deep Depression"
            alert_level = "CODE_YELLOW"
        else:
            category = "Low Pressure Area"
            alert_level = "CODE_GREEN"

        return {
            "category": category,
            "alert_level": alert_level,
            "vmax_knots": round(float(vmax_knots), 1),
            "vmax_kmh": round(float(vmax_kmh), 1),
            "pressure_deficit_hpa": round(float(delta_p), 2),
        }

    def process_timestep(self, time_index: int) -> Dict[str, Any]:
        """Process a single time-step to extract centroid, bbox, and intensity."""
        pressure_2d, _, _, ts_str = self.get_slice(time_index)
        centroid_lat, centroid_lon, min_p = self.extract_centroid(pressure_2d)
        bbox = self.extract_bounding_box(centroid_lat, centroid_lon)
        intensity = self.estimate_intensity(min_p)

        return {
            "time_index": time_index,
            "timestamp": ts_str,
            "centroid": {
                "lat": centroid_lat,
                "lon": centroid_lon,
            },
            "min_pressure_hpa": min_p,
            "bounding_box": bbox,  # [min_lon, min_lat, max_lon, max_lat]
            "intensity": intensity,
        }

    def track_all(self) -> List[Dict[str, Any]]:
        """Track cyclone trajectory across all available time steps."""
        track = []
        for idx in range(len(self.times)):
            track.append(self.process_timestep(idx))
        return track
