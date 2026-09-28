"""
scripts/train_diffusion.py
==========================
Train the Meteora Physics-Guided Diffusion Model on the real ERA5 Biparjoy data.

Usage:
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --epochs 100 --variable u10
    python scripts/train_diffusion.py --variable mslp --epochs 80 --physics-weight 0.1

The script:
  1. Loads biparjoy_real.nc (falls back to synthetic if not found)
  2. Creates (coarse 12km patch, fine 5km patch) training pairs by:
       - Taking each ERA5 timestep's spatial field
       - Creating a "coarse" version by downsampling 2.4x
       - Treating the original as the ground-truth HR target
  3. Trains ConditionalUNet with DDPM + Physics divergence penalty
  4. Saves the checkpoint to weights/meteora_diff_biparjoy.pt

After training, downscaler.py will automatically detect the checkpoint and
switch from the scipy mock to the real diffusion inference.

Estimated training time:
  CPU (i7/Ryzen 7): ~8–15 min for 50 epochs on 56 ERA5 timesteps
  Reduce --epochs to 20 for a quick smoke-test (<5 min).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "meteora_diff_biparjoy.pt"
)

SUPPORTED_VARIABLES = ["u10", "v10", "mslp", "precipitation"]


def make_pairs(
    ds,
    variable: str,
    coarse_sigma: float = 1.0,
) -> list:
    """
    Build (coarse_arr, fine_arr) training pairs from an xarray Dataset.

    Strategy:
        - 'fine' = the original ERA5 field at each timestep (ground truth)
        - 'coarse' = fine degraded by:
            1. Gaussian smoothing (simulates 12 km model resolution)
            2. Subsampling by 2.4x (reduce spatial resolution)
            3. Recovering to original size (like what a 12km model would see)

    This creates a self-supervised super-resolution dataset without needing
    a separate paired dataset.
    """
    from scipy.ndimage import gaussian_filter, zoom

    pairs = []
    n_times = ds.sizes["time"]

    for t in range(n_times):
        arr = ds[variable].isel(time=t).values.astype(np.float32)

        # Remove any NaNs (coastline land-mask artefacts)
        if np.any(np.isnan(arr)):
            arr = np.where(np.isnan(arr), np.nanmean(arr), arr)

        fine = arr  # ground truth HR

        # Create coarse: smooth + downsample + upsample (simulate 12 km)
        smoothed  = gaussian_filter(fine, sigma=coarse_sigma)
        scale_dn  = 1.0 / 2.4
        downsampled = zoom(smoothed, scale_dn, order=1)
        # Coarse is at reduced resolution; model will upsample from this
        pairs.append((downsampled, fine))

    print(f"  [pairs] variable='{variable}', n={len(pairs)}, "
          f"coarse_shape={pairs[0][0].shape}, fine_shape={pairs[0][1].shape}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variable",       default="u10",
                    choices=SUPPORTED_VARIABLES,
                    help="ERA5 variable to train on (default: u10)")
    ap.add_argument("--epochs",         type=int, default=50,
                    help="Number of training epochs (default: 50)")
    ap.add_argument("--batch-size",     type=int, default=4)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--physics-weight", type=float, default=0.05,
                    help="γ: physics divergence loss weight (default: 0.05)")
    ap.add_argument("--ddim-steps",     type=int, default=20,
                    help="DDIM sampling steps at inference (default: 20)")
    ap.add_argument("--out",            default=WEIGHTS_PATH,
                    help="Output checkpoint path")
    ap.add_argument("--no-physics",     action="store_true",
                    help="Disable physics loss (pure DDPM baseline)")
    ap.add_argument("--quick",          action="store_true",
                    help="Quick 10-epoch smoke test")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 10

    # ── Check PyTorch ──────────────────────────────────────────────────────
    try:
        import torch
        print(f"[torch] version={torch.__version__}  device=CPU")
    except ImportError:
        sys.exit(
            "PyTorch not found.\n"
            "Run: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    # ── Load data ──────────────────────────────────────────────────────────
    from core.data_loader import load_biparjoy_data, dataset_provenance

    print("\n[data] Loading Biparjoy dataset …")
    ds         = load_biparjoy_data()
    provenance = dataset_provenance(ds)
    print(f"  provenance : {provenance}")
    print(f"  time steps : {ds.sizes['time']}")
    print(f"  variables  : {list(ds.data_vars)}")
    print(f"  lat/lon    : {ds.sizes.get('lat','?')} x {ds.sizes.get('lon','?')}")

    if args.variable not in ds.data_vars:
        sys.exit(
            f"Variable '{args.variable}' not in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    # ── Build training pairs ───────────────────────────────────────────────
    print(f"\n[pairs] Building (coarse, fine) pairs for '{args.variable}' …")
    pairs = make_pairs(ds, args.variable)
    ds.close()

    if len(pairs) < 2:
        sys.exit("Too few training pairs. Check the dataset.")

    # ── Build model ────────────────────────────────────────────────────────
    from core.diffusion_model import MeteoraDiffusionModel

    model = MeteoraDiffusionModel(device="cpu")
    model.DDIM_S  = args.ddim_steps
    model.PHYS_W  = 0.0 if args.no_physics else args.physics_weight
    print(f"\n[model] Parameters: {model.parameter_count():,}")
    print(f"        DDIM steps : {model.DDIM_S}")
    print(f"        Physics w  : {model.PHYS_W}")
    print(f"        Epochs     : {args.epochs}")
    print(f"        Checkpoint : {args.out}\n")

    # ── Train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    print("=" * 60)
    print("TRAINING — Physics-Guided Diffusion Model")
    print("=" * 60)

    losses = model.train_on_dataset(
        pairs,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=model.PHYS_W,
        checkpoint_path=args.out,
    )

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Training complete in {elapsed/60:.1f} min")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Checkpoint: {args.out}")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Size      : {size_mb:.1f} MB")
    print("=" * 60)

    # ── Quick inference sanity check ───────────────────────────────────────
    print("\n[sanity] Running one DDIM inference pass...")
    coarse_test, fine_test = pairs[0]
    hr_out = model.ddim_sample(
        coarse_test, ddim_steps=5, physics_guidance=not args.no_physics
    )
    peak_preserved = abs(hr_out.max() - fine_test.max()) / (abs(fine_test.max()) + 1e-8) < 0.15
    print(f"  Coarse peak  : {coarse_test.max():.4f}")
    print(f"  Fine (truth) : {fine_test.max():.4f}")
    print(f"  PGDM output  : {hr_out.max():.4f}")
    print(f"  Peak within 15% of truth: {peak_preserved}")
    print("\n[OK] Model trained and ready for dashboard inference.")
    print("     The dashboard will auto-detect the checkpoint next time it loads.")


if __name__ == "__main__":
    main()
