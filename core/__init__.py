"""
Meteora Core Processing Modules.
Stage 1: Tracker (Centroid & Bounding Box extraction)
Stage 2: Downscaler (12 km -> 5 km diffusion/enhancement logic)
Stage 3: Physics (Mass & moisture flux verification stencil)
"""

from .tracker import CycloneTracker
from .downscaler import DiffusionDownscaler
from .physics import PhysicsVerifier

__all__ = ["CycloneTracker", "DiffusionDownscaler", "PhysicsVerifier"]
