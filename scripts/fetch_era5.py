"""
Download real ERA5 reanalysis for Cyclone Biparjoy (June 2023) from the
Copernicus Climate Data Store, then normalise it into the canonical Meteora
schema.

Requires a free CDS account and a configured ~/.cdsapirc.
See README.md -> "Getting the real data" for setup.

Usage:
    python scripts/fetch_era5.py                 # 6-hourly (default, small)
    python scripts/fetch_era5.py --hourly        # hourly (bigger, finer)
    python scripts/fetch_era5.py --start 2023-06-06 --end 2023-06-19
"""

from __future__ import annotations

import argparse
import os
import sys

# Bounding box in CDS order: North, West, South, East.
# Covers the Arabian Sea, the Biparjoy track, and the Gujarat landfall coast.
AREA = [30.0, 55.0, 8.0, 78.0]

# ERA5 short names -> the variables Stage 1/2 actually consume.
VARIABLES = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

DATASET = "reanalysis-era5-single-levels"


def build_request(start: str, end: str, hourly: bool) -> dict:
    """Build the CDS request dict, expanding the date range into Y/M/D lists."""
    import pandas as pd

    dates = pd.date_range(start, end, freq="D")
    if dates.empty:
        sys.exit(f"Empty date range: {start} -> {end}")

    years = sorted({d.strftime("%Y") for d in dates})
    months = sorted({d.strftime("%m") for d in dates})
    if len(months) > 1:
        sys.exit(
            "This script assumes a single calendar month per request "
            f"(got {months}). Split your range and request the months separately."
        )

    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": years,
        "month": months,
        "day": sorted({d.strftime("%d") for d in dates}),
        "time": (
            [f"{h:02d}:00" for h in range(24)]
            if hourly
            else ["00:00", "06:00", "12:00", "18:00"]
        ),
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-06-06", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default="2023-06-19", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--hourly", action="store_true", help="all 24 hours instead of 4")
    ap.add_argument(
        "--out", default="data/era5_biparjoy_2023_raw.nc", help="output NetCDF path"
    )
    args = ap.parse_args()

    try:
        import cdsapi
    except ImportError:
        sys.exit("cdsapi is not installed.  Run:  pip install cdsapi")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out):
        print(f"[skip] {args.out} already exists. Delete it to re-download.")
        return

    request = build_request(args.start, args.end, args.hourly)
    n_times = len(request["day"]) * len(request["time"])
    print(f"[CDS] {DATASET}")
    print(f"      range   : {args.start} -> {args.end}")
    print(f"      grid    : {len(request['day'])} days x {len(request['time'])} hours"
          f"  =  {n_times} timesteps")
    print(f"      area    : N{AREA[0]} W{AREA[1]} S{AREA[2]} E{AREA[3]}")
    print(f"      vars    : {', '.join(VARIABLES)}")
    print("      (CDS queues requests; this typically takes 1-5 minutes)\n")

    client = cdsapi.Client()
    client.retrieve(DATASET, request, args.out)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n[OK] downloaded -> {args.out}  ({size_mb:.1f} MB)")
    print("Next:  python scripts/build_dataset.py   # normalise into Meteora schema")


if __name__ == "__main__":
    main()
