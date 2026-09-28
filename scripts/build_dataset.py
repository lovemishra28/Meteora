"""
Normalise a raw ERA5 download into the canonical Meteora dataset.

    python scripts/build_dataset.py
    python scripts/build_dataset.py --raw data/era5_biparjoy_2023_raw.nc

Reads the raw CDS file, applies unit/coordinate normalisation and
de-accumulation (see core/data_loader.normalise_era5), writes
data/biparjoy_real.nc, then prints a schema report so the result can be
eyeballed before anything downstream consumes it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import (  # noqa: E402
    RAW_ERA5,
    REAL_CANONICAL,
    normalise_era5,
    open_raw_era5,
)


def report(ds: xr.Dataset) -> None:
    """Print the schema and a physical sanity check on the normalised field."""
    print("\n" + "=" * 68)
    print("CANONICAL DATASET")
    print("=" * 68)
    print(f"dimensions : {dict(ds.sizes)}")
    print(f"variables  : {sorted(ds.data_vars)}")
    print(f"time       : {str(ds.time.values[0])[:16]} -> {str(ds.time.values[-1])[:16]}"
          f"  ({ds.sizes['time']} steps)")
    print(f"lat        : {float(ds.lat.min()):.2f} .. {float(ds.lat.max()):.2f} "
          f"({'ascending' if float(ds.lat[0]) < float(ds.lat[-1]) else 'DESCENDING!'})")
    print(f"lon        : {float(ds.lon.min()):.2f} .. {float(ds.lon.max()):.2f}")

    # Physical sanity: a real tropical cyclone must show a closed pressure
    # minimum, and that minimum must migrate over time -- not sit still, and
    # not sit at a domain edge (which would mean the storm is outside the box).
    print("\nper-timestep minimum pressure (cyclone eye):")
    for i in range(ds.sizes["time"]):
        sl = ds.isel(time=i)
        idx = np.unravel_index(np.argmin(sl["mslp"].values), sl["mslp"].shape)
        lat, lon = float(ds.lat.values[idx[0]]), float(ds.lon.values[idx[1]])
        wind = np.sqrt(sl["u10"].values**2 + sl["v10"].values**2).max()
        edge = (
            idx[0] in (0, ds.sizes["lat"] - 1) or idx[1] in (0, ds.sizes["lon"] - 1)
        )
        flag = "  <-- AT DOMAIN EDGE: storm may be clipped!" if edge else ""
        print(
            f"  t={i:2d} {str(ds.time.values[i])[:16]}  "
            f"mslp={float(sl['mslp'].min()):7.2f} hPa  "
            f"eye=({lat:6.2f}, {lon:7.2f})  "
            f"max|V|={wind:6.1f} km/h{flag}"
        )

    print("\n" + "=" * 68)
    print("If the eye moves smoothly and the pressure minimum is in the")
    print("50-1000 hPa range, this is real Biparjoy and we can proceed.")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=RAW_ERA5)
    ap.add_argument("--out", default=REAL_CANONICAL)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the intermediate normalised file (default: yes)")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        sys.exit(f"Raw file not found: {args.raw}\nRun scripts/fetch_era5.py first.")

    print(f"[build] reading  {args.raw}")
    raw = open_raw_era5(args.raw)
    print(f"[build] raw keys : {sorted(raw.data_vars)}")
    print(f"[build] raw dims : {dict(raw.sizes)}")

    ds = normalise_era5(raw)
    raw.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds.to_netcdf(args.out)
    print(f"[build] wrote    {args.out}")

    reportds = xr.open_dataset(args.out)
    report(reportds)
    reportds.close()


if __name__ == "__main__":
    main()
