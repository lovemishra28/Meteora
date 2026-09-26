"""
Meteora Core Processing Modules.
Stage 1: Tracker (Centroid & Bounding Box extraction)
Stage 2: Downscaler (12 km -> 5 km diffusion/enhancement logic)
Stage 3: Physics (Mass & moisture flux verification stencil)
"""

from .data_loader import generate_biparjoy_data, load_biparjoy_data
from .tracker import (
    AnomalyTracker,
    CycloneTracker,
    create_sample_biparjoy_netcdf,
    extract_anomaly_bounding_box,
)
from .downscaler import DownscalerEngine, DiffusionDownscaler
from .physics import PhysicsVerifier

__all__ = [
    "AnomalyTracker",
    "generate_biparjoy_data",
    "load_biparjoy_data",
    "CycloneTracker",
    "create_sample_biparjoy_netcdf",
    "extract_anomaly_bounding_box",
    "DownscalerEngine",
    "DiffusionDownscaler",
    "PhysicsVerifier",
]
