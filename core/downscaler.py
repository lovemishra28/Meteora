"""
Stage 2: Downscaler - 12 km -> 5 km diffusion & enhancement logic.
Super-resolves atmospheric pressure fields using physics-guided diffusion
relaxation and gradient sharpening to reconstruct sharp cyclone eyewalls
and convective rainbands.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, laplace


class DiffusionDownscaler:
    """Stage 2: 12 km -> 5 km Diffusion and Enhancement Model."""

    def __init__(
        self,
        coarse_res_km: float = 12.0,
        target_res_km: float = 5.0,
        diffusion_steps: int = 10,
        diffusion_rate: float = 0.12,
        sharpening_strength: float = 0.35,
    ):
        """
        :param coarse_res_km: Nominal coarse resolution (12 km).
        :param target_res_km: Target super-resolved resolution (5 km).
        :param diffusion_steps: Number of diffusion refinement iterations.
        :param diffusion_rate: Numerical PDE step size alpha.
        :param sharpening_strength: Physics gradient enhancement magnitude.
        """
        self.coarse_res_km = coarse_res_km
        self.target_res_km = target_res_km
        self.scale_factor = coarse_res_km / target_res_km  # ~2.4x
        self.diffusion_steps = diffusion_steps
        self.diffusion_rate = diffusion_rate
        self.sharpening_strength = sharpening_strength

    def build_high_res_coords(
        self,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate high-resolution 1D latitude and longitude arrays
        targeting ~5 km grid spacing (~0.045 degrees).
        """
        lat_step_deg = self.target_res_km / 111.0  # ~0.045 deg
        lon_step_deg = self.target_res_km / (111.0 * np.cos(np.radians(np.mean(lats))))

        num_lats = max(int(np.ceil((lats.max() - lats.min()) / lat_step_deg)) + 1, int(len(lats) * self.scale_factor))
        num_lons = max(int(np.ceil((lons.max() - lons.min()) / lon_step_deg)) + 1, int(len(lons) * self.scale_factor))

        hr_lats = np.linspace(lats.min(), lats.max(), num_lats)
        hr_lons = np.linspace(lons.min(), lons.max(), num_lons)
        return hr_lats, hr_lons

    def initial_bicubic_interpolation(
        self,
        coarse_field: np.ndarray,
        coarse_lats: np.ndarray,
        coarse_lons: np.ndarray,
        hr_lats: np.ndarray,
        hr_lons: np.ndarray
    ) -> np.ndarray:
        """High-order regular grid spline interpolation as initial condition."""
        interp = RegularGridInterpolator(
            (coarse_lats, coarse_lons),
            coarse_field,
            method="cubic",
            bounds_error=False,
            fill_value=None
        )
        mesh_lat, mesh_lon = np.meshgrid(hr_lats, hr_lons, indexing="ij")
        points = np.stack([mesh_lat.ravel(), mesh_lon.ravel()], axis=-1)
        hr_field = interp(points).reshape(len(hr_lats), len(hr_lons))
        return hr_field

    def physics_diffusion_step(
        self,
        field: np.ndarray,
        base_field: np.ndarray,
        centroid_idx: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Single step of reverse diffusion relaxation with physics-based sharpening.
        Reconstructs non-linear pressure drop in eyewall and spiral band perturbations.
        """
        # Discrete Laplacian operator
        lap = laplace(field)

        # Gradient magnitude for selective sharpening near eyewall
        gy, gx = np.gradient(field)
        grad_mag = np.sqrt(gx**2 + gy**2)
        norm_grad = grad_mag / (np.max(grad_mag) + 1e-6)

        # Eyewall enhancement stencil (sharpen where gradient is steep)
        shock_term = -self.sharpening_strength * (norm_grad * lap)

        # Restoration term to anchor against coarse-scale drift
        restoration = -0.15 * (field - base_field)

        # Update PDE step
        updated = field + self.diffusion_rate * (shock_term + restoration)

        # Enforce global mean pressure conservation
        bias = np.mean(updated) - np.mean(base_field)
        updated = updated - bias

        return updated

    def downscale(
        self,
        coarse_field: np.ndarray,
        coarse_lats: np.ndarray,
        coarse_lons: np.ndarray,
        apply_diffusion: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 Downscaling: Coarse (12 km) -> Enhanced High-Res (5 km).

        :return: Dict containing enhanced field, high-res grid, and metadata.
        """
        hr_lats, hr_lons = self.build_high_res_coords(coarse_lats, coarse_lons)
        hr_base = self.initial_bicubic_interpolation(
            coarse_field, coarse_lats, coarse_lons, hr_lats, hr_lons
        )

        hr_field = hr_base.copy()
        if apply_diffusion:
            for _ in range(self.diffusion_steps):
                hr_field = self.physics_diffusion_step(hr_field, hr_base)

        # Calculate sharpening and gradient improvements in physical units (hPa/deg)
        coarse_gy, coarse_gx = np.gradient(coarse_field, coarse_lats, coarse_lons)
        hr_gy, hr_gx = np.gradient(hr_field, hr_lats, hr_lons)

        max_grad_coarse = float(np.max(np.sqrt(coarse_gx**2 + coarse_gy**2)))
        max_grad_hr = float(np.max(np.sqrt(hr_gx**2 + hr_gy**2)))

        return {
            "hr_field": hr_field,
            "hr_base": hr_base,
            "hr_lats": hr_lats,
            "hr_lons": hr_lons,
            "coarse_shape": coarse_field.shape,
            "hr_shape": hr_field.shape,
            "coarse_res_km": self.coarse_res_km,
            "target_res_km": self.target_res_km,
            "diffusion_steps_applied": self.diffusion_steps if apply_diffusion else 0,
            "max_gradient_coarse": round(max_grad_coarse, 3),
            "max_gradient_hr": round(max_grad_hr, 3),
            "gradient_enhancement_ratio": round(max_grad_hr / (max_grad_coarse + 1e-6), 2),
        }
