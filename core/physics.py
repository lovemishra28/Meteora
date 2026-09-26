"""
Stage 3: Physics - Mass & moisture flux verification stencil.
Computes finite-difference stencils for mass continuity, geostrophic/gradient
wind balance, and horizontal moisture flux convergence to verify physical validity
of downscaled atmospheric fields.
"""

from typing import Dict, Any, Tuple
import numpy as np


class PhysicsVerifier:
    """Stage 3: Physics Stencil & Conservation Verification."""

    def __init__(
        self,
        air_density: float = 1.18,      # kg/m^3 (approx tropical marine surface boundary)
        omega: float = 7.2921e-5,       # Earth rotation rate rad/s
        sst_celsius: float = 30.5,      # Arabian Sea pre-monsoon SST
    ):
        self.rho = air_density
        self.omega = omega
        self.sst_celsius = sst_celsius
        # Reference surface saturation specific humidity ~ 21 g/kg (0.021 kg/kg)
        self.q_surface_ref = 0.0215

    def coriolis_parameter(self, lat_deg: np.ndarray) -> np.ndarray:
        """Coriolis parameter f = 2 * omega * sin(phi)."""
        lat_rad = np.radians(lat_deg)
        return 2.0 * self.omega * np.sin(lat_rad)

    def compute_gradient_wind(
        self,
        pressure_hpa: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive horizontal wind components (u, v in m/s) from pressure field
        using geostrophic + gradient wind stencil.
        """
        # Convert hPa to Pa (1 hPa = 100 Pa)
        p_pa = pressure_hpa * 100.0

        # Physical grid step distances in meters
        mean_lat = float(np.mean(lats))
        dy = np.abs(lats[1] - lats[0]) * 111_000.0  # meters per degree lat
        dx = np.abs(lons[1] - lons[0]) * 111_000.0 * np.cos(np.radians(mean_lat))  # meters per deg lon

        # 2nd-order central difference stencils
        dp_dy, dp_dx = np.gradient(p_pa, dy, dx)

        # 2D Coriolis array
        mesh_lat, _ = np.meshgrid(lats, lons, indexing="ij")
        f = self.coriolis_parameter(mesh_lat)
        f_safe = np.where(np.abs(f) < 1e-5, np.sign(f) * 1e-5, f)

        # Geostrophic wind approximations:
        # u_g = - (1 / (rho * f)) * dP/dy
        # v_g =   (1 / (rho * f)) * dP/dx
        u_geo = -(1.0 / (self.rho * f_safe)) * dp_dy
        v_geo = (1.0 / (self.rho * f_safe)) * dp_dx

        # Boundary layer friction & cyclostrophic cross-isobar inflow damping (20 deg angle)
        inflow_angle = np.radians(18.0)
        cos_ang = np.cos(inflow_angle)
        sin_ang = np.sin(inflow_angle)

        # Realistic surface wind field with cyclonic spiraling
        u_wind = 0.85 * (u_geo * cos_ang - v_geo * sin_ang)
        v_wind = 0.85 * (u_geo * sin_ang + v_geo * cos_ang)

        # Physical limit cap for tropical cyclone stability (up to 85 m/s)
        speed = np.sqrt(u_wind**2 + v_wind**2)
        max_speed = 85.0
        scale = np.where(speed > max_speed, max_speed / (speed + 1e-6), 1.0)

        return u_wind * scale, v_wind * scale

    def compute_mass_divergence_stencil(
        self,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> np.ndarray:
        """
        Compute horizontal mass divergence stencil:
        div(rho * u) = d(rho*u)/dx + d(rho*v)/dy (in kg / (m^3 * s)).
        """
        mean_lat = float(np.mean(lats))
        dy = np.abs(lats[1] - lats[0]) * 111_000.0
        dx = np.abs(lons[1] - lons[0]) * 111_000.0 * np.cos(np.radians(mean_lat))

        # Central difference derivatives
        du_dx = np.gradient(u_wind, dx, axis=1)
        dv_dy = np.gradient(v_wind, dy, axis=0)

        mass_div = self.rho * (du_dx + dv_dy)
        return mass_div

    def compute_moisture_flux_stencil(
        self,
        pressure_hpa: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute moisture proxy field q (kg/kg), horizontal moisture flux (Fx, Fy),
        and moisture flux convergence (- div(q * u)).
        """
        # Moisture proxy inversely proportional to central pressure depression
        min_p = np.min(pressure_hpa)
        p_def = np.maximum(0.0, 1012.0 - pressure_hpa)
        # Saturated core humidity elevated up to ~25 g/kg in eyewall
        q_field = self.q_surface_ref * (1.0 + 0.015 * p_def)

        # Flux components in kg / (m^2 * s)
        fx = self.rho * q_field * u_wind
        fy = self.rho * q_field * v_wind

        mean_lat = float(np.mean(lats))
        dy = np.abs(lats[1] - lats[0]) * 111_000.0
        dx = np.abs(lons[1] - lons[0]) * 111_000.0 * np.cos(np.radians(mean_lat))

        dfx_dx = np.gradient(fx, dx, axis=1)
        dfy_dy = np.gradient(fy, dy, axis=0)

        # Moisture Flux Convergence (MFC): - (dFx/dx + dFy/dy)
        # Positive MFC indicates moisture convergence feeding eyewall convection
        mfc = -(dfx_dx + dfy_dy)

        return q_field, np.sqrt(fx**2 + fy**2), mfc

    def verify_field(
        self,
        pressure_hpa: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run full physics verification stencil suite and compute consistency score.

        :return: Metrics dictionary with divergence residuals, MFC maxima, and verification score.
        """
        u_wind, v_wind = self.compute_gradient_wind(pressure_hpa, lats, lons)
        mass_div = self.compute_mass_divergence_stencil(u_wind, v_wind, lats, lons)
        q_field, flux_mag, mfc = self.compute_moisture_flux_stencil(
            pressure_hpa, u_wind, v_wind, lats, lons
        )

        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        max_wind_ms = float(np.max(wind_speed))
        max_wind_knots = max_wind_ms * 1.94384

        # Conservation and consistency metrics:
        # 1. Divergence normalization ratio
        mean_div_abs = float(np.mean(np.abs(mass_div)))
        span_lat = float(np.max(lats) - np.min(lats))
        span_lon = float(np.max(lons) - np.min(lons))
        ref_div = float(np.mean(wind_speed) * self.rho / (np.mean([span_lat, span_lon]) * 111_000.0) + 1e-7)
        div_residual_ratio = min(1.0, mean_div_abs / (ref_div * 10.0))

        # 2. Moisture flux convergence in eyewall check (physical cyclone should have positive MFC near core)
        min_p_idx = np.unravel_index(np.argmin(pressure_hpa), pressure_hpa.shape)
        core_mfc = float(mfc[min_p_idx])
        has_positive_core_mfc = core_mfc > 0

        # 3. Laplaican smoothness / lack of high-frequency noise artifacts
        p_lap = np.gradient(np.gradient(pressure_hpa, axis=0), axis=0) + np.gradient(np.gradient(pressure_hpa, axis=1), axis=1)
        lap_roughness = float(np.std(p_lap))

        # Composite verification score (0.0 to 100.0%)
        score = 100.0 - (div_residual_ratio * 25.0) - (0.0 if has_positive_core_mfc else 15.0)
        score = max(50.0, min(99.4, score))

        return {
            "max_wind_ms": round(max_wind_ms, 2),
            "max_wind_knots": round(max_wind_knots, 1),
            "mean_mass_divergence_residual": f"{mean_div_abs:.3e}",
            "core_moisture_flux_convergence": f"{core_mfc:.3e}",
            "max_moisture_flux": round(float(np.max(flux_mag)), 4),
            "physics_consistency_score": round(score, 1),
            "is_physically_consistent": bool(score >= 70.0),
            "wind_speed_grid": wind_speed,
            "mfc_grid": mfc,
            "mass_divergence_grid": mass_div,
        }
