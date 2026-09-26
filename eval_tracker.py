"""
Evaluation script for Stage 1: Data & Tracking Mock Engine.
Validates synthetic NetCDF dataset generation, cyclone pressure drop identification,
and Stage 1 bounding box extraction across all simulated time intervals.
"""

import os
import sys
import time
import xarray as xr
import numpy as np

# Ensure local packages are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.tracker import (
    create_sample_biparjoy_netcdf,
    extract_anomaly_bounding_box,
    CycloneTracker
)


# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def evaluate_stage1_tracker(nc_path: str = "data/biparjoy_sample.nc"):
    print("=" * 75)
    print("[METEORA] STAGE 1: DATA & TRACKING MOCK ENGINE EVALUATION")
    print("=" * 75)

    # 1. Generate / Verify Dataset
    print(f"\n[1/3] Generating synthetic Cyclone Biparjoy NetCDF dataset -> '{nc_path}'...")
    start_gen = time.perf_counter()
    create_sample_biparjoy_netcdf(filepath=nc_path)
    gen_time = (time.perf_counter() - start_gen) * 1000
    print(f"      Generation completed in {gen_time:.2f} ms")

    # 2. Inspect Dataset Structure
    print("\n[2/3] Inspecting NetCDF metadata and dimensional coordinates...")
    ds = xr.open_dataset(nc_path)
    print(f"      Dimensions: {dict(ds.sizes)}")
    print(f"      Data Variables: {list(ds.data_vars.keys())}")
    print(f"      Time steps: {len(ds.time)} ({str(ds.time.values[0])[:16]} to {str(ds.time.values[-1])[:16]})")
    print(f"      Lat Range: [{float(ds.lat.min()):.1f}, {float(ds.lat.max()):.1f}] ({len(ds.lat)} points)")
    print(f"      Lon Range: [{float(ds.lon.min()):.1f}, {float(ds.lon.max()):.1f}] ({len(ds.lon)} points)")

    # Ground truth centers used in mock simulation
    expected_centers = [
        (18.0, 67.5), (19.2, 67.8), (20.5, 68.2), 
        (21.8, 68.7), (22.8, 69.3), (23.5, 70.1)
    ]

    # 3. Evaluate Anomaly Bounding Box Extraction
    print("\n[3/3] Evaluating extract_anomaly_bounding_box across all timesteps:")
    print("-" * 75)
    print(f"{'Step':<5} | {'Timestamp':<16} | {'True Center':<14} | {'Detected Centroid':<17} | {'Error':<7} | {'MSLP Min':<9}")
    print("-" * 75)

    all_passed = True
    tracker = CycloneTracker(nc_path)

    for t_idx in range(len(ds.time)):
        bbox_res = extract_anomaly_bounding_box(ds, time_idx=t_idx, threshold_hpa=990.0)
        
        assert bbox_res is not None, f"Failed: No anomaly detected at timestep {t_idx}!"
        
        centroid = bbox_res["centroid"]
        bounds = bbox_res["bounds"]
        true_lat, true_lon = expected_centers[t_idx]
        det_lat, det_lon = centroid

        # Compute euclidean distance in degrees
        dist_err = np.sqrt((det_lat - true_lat)**2 + (det_lon - true_lon)**2)
        
        # Min pressure at this slice
        slice_p = ds["mslp"].isel(time=t_idx)
        min_p = float(slice_p.min())

        ts_short = bbox_res["time"][:16]
        true_str = f"({true_lat:.1f}, {true_lon:.1f})"
        det_str = f"({det_lat:.2f}, {det_lon:.2f})"
        err_str = f"{dist_err:.3f}°"
        p_str = f"{min_p:.1f} hPa"

        print(f"{t_idx:<5} | {ts_short:<16} | {true_str:<14} | {det_str:<17} | {err_str:<7} | {p_str:<9}")

        # Assertions
        assert dist_err < 0.25, f"Centroid error too large: {dist_err:.3f} deg at step {t_idx}"
        assert bounds["lat_min"] < bounds["lat_max"], "Invalid latitude bounds!"
        assert bounds["lon_min"] < bounds["lon_max"], "Invalid longitude bounds!"
        assert (bounds["lat_max"] - bounds["lat_min"]) == 3.0, "Bounding box latitude span should be 3.0 deg"
        assert (bounds["lon_max"] - bounds["lon_min"]) == 3.0, "Bounding box longitude span should be 3.0 deg"

    print("-" * 75)

    # Validate CycloneTracker class integration
    print("\n[BONUS] Verifying CycloneTracker class end-to-end integration...")
    full_track = tracker.track_all()
    print(f"      Tracked {len(full_track)} timesteps successfully.")
    first_step = full_track[0]
    print(f"      Step 0 Category: {first_step['intensity']['category']} (Alert: {first_step['intensity']['alert_level']})")
    print(f"      Step 0 Wind Speed: {first_step['intensity']['vmax_knots']} kts ({first_step['intensity']['vmax_kmh']} km/h)")
    print(f"      Step 0 Bounding Box: {first_step['bounding_box']}")

    ds.close()
    print("\n" + "=" * 75)
    print("[SUCCESS] ALL CHECKS PASSED: Stage 1 Mock Engine & Tracker operational!")
    print("=" * 75)


if __name__ == "__main__":
    # If run from meteora-prototype directory or repo root:
    base_data_path = os.path.join(CURRENT_DIR, "data", "biparjoy_sample.nc")
    evaluate_stage1_tracker(base_data_path)
