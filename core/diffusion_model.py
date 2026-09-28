# core/diffusion_model.py
"""
Meteora Physics-Guided Diffusion Model (PGDM)
==============================================

Architecture:   Lightweight 2-D Conditional U-Net (~2.4 M parameters)
Noise schedule: Cosine beta schedule (DDPM)
Sampler:        DDIM (10–20 steps, deterministic, fast at inference)
Physics Guide:  Mass-divergence penalty injected at every DDIM step

Task: 12 km coarse grid  ->  5 km super-resolved grid
      scale factor = 2.4x  (same as original DownscalerEngine)

Inference path (used by downscaler.py):
    model = MeteoraDiffusionModel.load("weights/meteora_diff_biparjoy.pt")
    hr    = model.ddim_sample(coarse_patch, physics_guidance=True)

Training path (used by scripts/train_diffusion.py):
    model.train_on_dataset(dataset, n_epochs=50)
    model.save("weights/meteora_diff_biparjoy.pt")
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# -----------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021)
# -----------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008):
    """Cosine schedule as proposed in 'Improved DDPM' (Nichol & Dhariwal 2021)."""
    steps = T + 1
    t = np.linspace(0, T, steps) / T
    alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = np.clip(betas, 0, 0.999)
    return betas


# -----------------------------------------------------------------------------
# Building blocks: ResBlock, Attention, U-Net
# -----------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SinusoidalTimeEmbedding(nn.Module):
        """Sinusoidal timestep encoding -- same as the original DDPM paper."""
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
            ).float()
            args  = t[:, None].float() * freqs[None]
            return torch.cat([args.sin(), args.cos()], dim=-1)

    class ResBlock(nn.Module):
        """Residual block with GroupNorm + time-embedding injection."""
        def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
            super().__init__()
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
            self.block1   = nn.Sequential(
                nn.GroupNorm(groups, in_ch), nn.SiLU(),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
            self.block2   = nn.Sequential(
                nn.GroupNorm(groups, out_ch), nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            )
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
            h = self.block1(x)
            h = h + self.time_mlp(t_emb)[:, :, None, None]
            h = self.block2(h)
            return h + self.res_conv(x)

    class Attention2D(nn.Module):
        """Lightweight self-attention on spatial feature maps."""
        def __init__(self, ch: int, heads: int = 4):
            super().__init__()
            self.heads = heads
            self.norm  = nn.GroupNorm(1, ch)
            self.qkv   = nn.Conv2d(ch, ch * 3, 1)
            self.proj  = nn.Conv2d(ch, ch, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            h = self.norm(x)
            qkv = self.qkv(h).chunk(3, dim=1)
            q, k, v = [t.reshape(B, self.heads, C // self.heads, H * W) for t in qkv]
            scale = (C // self.heads) ** -0.5
            attn  = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
            out   = (attn @ v).reshape(B, C, H, W)
            return x + self.proj(out)

    class ConditionalUNet(nn.Module):
        """
        Lightweight 2-D conditional U-Net.

        Conditioning: the coarse 12 km field is bilinearly upsampled to the
        target HR size and concatenated channel-wise with the noisy HR field,
        giving the model explicit access to the coarse structure at every layer.

        Channels: 64 -> 128 -> 256 (encoder), mirrored decoder.
        Parameters: ~2.4 M (fast CPU inference, fits in 512 MB RAM).
        """
        def __init__(
            self,
            in_channels: int  = 2,   # noisy HR + upsampled coarse (1+1)
            out_channels: int = 1,
            base_ch: int      = 64,
            time_dim: int     = 128,
        ):
            super().__init__()
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(base_ch),
                nn.Linear(base_ch, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            ch = [base_ch, base_ch * 2, base_ch * 4]  # [64, 128, 256]

            # -- Encoder ----------------------------------------------------
            self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)
            self.down1     = ResBlock(ch[0], ch[0], time_dim)
            self.down2     = nn.Sequential(
                nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1),
            )
            self.down3     = ResBlock(ch[1], ch[1], time_dim, groups=8)
            self.down4     = nn.Sequential(
                nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1),
            )

            # -- Bottleneck with attention -----------------------------------
            self.mid1  = ResBlock(ch[2], ch[2], time_dim, groups=8)
            self.attn  = Attention2D(ch[2])
            self.mid2  = ResBlock(ch[2], ch[2], time_dim, groups=8)

            # -- Decoder ----------------------------------------------------
            self.up4   = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
            self.up3   = ResBlock(ch[1] * 2, ch[1], time_dim, groups=8)
            self.up2   = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
            self.up1   = ResBlock(ch[0] * 2, ch[0], time_dim)

            self.out_conv = nn.Sequential(
                nn.GroupNorm(8, ch[0]),
                nn.SiLU(),
                nn.Conv2d(ch[0], out_channels, 1),
            )

        def forward(
            self,
            x: torch.Tensor,        # (B, 1, H, W)  noisy HR
            t: torch.Tensor,        # (B,)           timestep indices
            cond: torch.Tensor,     # (B, 1, H, W)  upsampled coarse field
        ) -> torch.Tensor:
            t_emb = self.time_embed(t)
            # Concatenate conditioning
            h0    = self.init_conv(torch.cat([x, cond], dim=1))

            # Encoder
            h1 = self.down1(h0, t_emb)          # (B, 64, H, W)
            h2 = self.down3(self.down2(h1), t_emb)   # (B, 128, H/2, W/2)
            h3 = self.down4(h2)                  # (B, 256, H/4, W/4)

            # Bottleneck
            h3 = self.mid2(self.attn(self.mid1(h3, t_emb)), t_emb)

            # Decoder with skip connections
            # F.interpolate aligns dims for non-power-of-2 grids (e.g., 37x39)
            up4_out = F.interpolate(self.up4(h3), size=h2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.up3(torch.cat([up4_out, h2], dim=1), t_emb)
            up2_out = F.interpolate(self.up2(d2), size=h1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.up1(torch.cat([up2_out, h1], dim=1), t_emb)

            return self.out_conv(d1)


# -----------------------------------------------------------------------------
# Physics guidance: mass-divergence penalty
# -----------------------------------------------------------------------------

    def _mass_divergence_loss(
        field: torch.Tensor,    # (B, 1, H, W)
        dx: float = 5000.0,     # 5 km grid spacing in metres
    ) -> torch.Tensor:
        """
        Compute L2 norm of horizontal mass divergence as a physics penalty.

        For a scalar wind-proxy field we use ∇²ψ (Laplacian) as a proxy for
        divergence magnitude -- zero Laplacian = non-divergent irrotational flow.

        Returns scalar loss (sum over batch).
        """
        # 2-D Laplacian via finite-difference convolution
        laplacian_kernel = torch.tensor(
            [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]]
        , dtype=field.dtype, device=field.device) / (dx ** 2)
        lap = F.conv2d(field, laplacian_kernel, padding=1)
        return (lap ** 2).mean()


# -----------------------------------------------------------------------------
# Main wrapper: MeteoraDiffusionModel
# -----------------------------------------------------------------------------

class MeteoraDiffusionModel:
    """
    High-level wrapper around ConditionalUNet + DDPM/DDIM.

    Public API
    ----------
    MeteoraDiffusionModel.load(path)             -> model
    model.ddim_sample(coarse_np, ...)            -> hr_np (numpy)
    model.train_on_dataset(pairs, n_epochs, ...)
    model.save(path)
    """

    # Training hyper-parameters
    T       : int   = 1000     # Total diffusion steps
    DDIM_S  : int   = 20       # DDIM inference steps
    LR      : float = 1e-4
    PHYS_W  : float = 0.05     # Physics guidance weight (γ)
    SCALE   : float = 2.4      # 12 km -> 5 km

    def __init__(self, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed.\n"
                "Run:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        self.device = torch.device(device)

        # Build noise schedule
        betas = cosine_beta_schedule(self.T)
        self.betas              = torch.tensor(betas, dtype=torch.float32)
        alphas                  = 1.0 - self.betas
        self.alphas_cumprod     = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod        = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas      = torch.sqrt(1.0 - self.alphas_cumprod)

        # Build model
        self.net = ConditionalUNet().to(self.device)
        self._trained = False

    # -- Noise schedule helpers ---------------------------------------------

    def _q_sample(
        self,
        x0: "torch.Tensor",
        t:  "torch.Tensor",
        noise: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].to(self.device)[:, None, None, None]
        sqrt_b = self.sqrt_one_minus_alphas[t].to(self.device)[:, None, None, None]
        return sqrt_a * x0 + sqrt_b * noise

    # -- Training ----------------------------------------------------------

    def train_on_dataset(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
        n_epochs: int = 50,
        batch_size: int = 4,
        lr: float = None,
        physics_weight: float = None,
        checkpoint_path: str = "weights/meteora_diff_biparjoy.pt",
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the model on (coarse_12km, fine_5km) numpy array pairs.

        Parameters
        ----------
        pairs       : list of (coarse_arr, fine_arr) with same spatial shape
                      after coarse is upsampled.
        n_epochs    : training epochs.
        physics_weight : γ -- weight of the physics (divergence) loss term.

        Returns
        -------
        List of per-epoch average losses.
        """
        lr            = lr or self.LR
        physics_weight = physics_weight or self.PHYS_W

        import torch.optim as optim
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        epoch_losses = []

        self.net.train()
        for epoch in range(n_epochs):
            np.random.shuffle(pairs)  # type: ignore[arg-type]
            batch_losses = []

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                # Build tensors
                x0_list, cond_list = [], []
                for coarse, fine in batch:
                    # Normalise to [-1, 1]
                    p_min, p_max = fine.min(), fine.max()
                    fine_n   = 2 * (fine  - p_min) / (p_max - p_min + 1e-8) - 1
                    coarse_n = 2 * (coarse - p_min) / (p_max - p_min + 1e-8) - 1

                    # Upsample coarse to HR size via bilinear
                    coarse_t = torch.tensor(coarse_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    fine_t   = torch.tensor(fine_n,   dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    target_size = fine_t.shape[-2:]
                    coarse_up   = F.interpolate(coarse_t, size=target_size, mode="bilinear", align_corners=False)
                    x0_list.append(fine_t.squeeze(0))
                    cond_list.append(coarse_up.squeeze(0))

                x0   = torch.stack(x0_list).to(self.device)
                cond = torch.stack(cond_list).to(self.device)
                B    = x0.shape[0]

                # Sample random timesteps
                t_idx = torch.randint(0, self.T, (B,))
                noise = torch.randn_like(x0)
                x_t   = self._q_sample(x0, t_idx, noise)

                # Predict noise
                t_dev  = t_idx.to(self.device)
                pred_n = self.net(x_t, t_dev, cond)

                # DDPM denoising loss
                diffusion_loss = F.mse_loss(pred_n, noise)

                # Physics penalty on predicted x0
                alpha_t  = self.alphas_cumprod[t_idx].to(self.device)[:, None, None, None]
                sqrt_at  = torch.sqrt(alpha_t)
                sqrt_1mt = torch.sqrt(1.0 - alpha_t)
                x0_pred  = (x_t - sqrt_1mt * pred_n) / (sqrt_at + 1e-8)
                phys_loss = _mass_divergence_loss(x0_pred)

                loss = diffusion_loss + physics_weight * phys_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            scheduler.step()
            avg = float(np.mean(batch_losses))
            epoch_losses.append(avg)
            if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg:.6f}")

        self._trained = True
        self.save(checkpoint_path)
        return epoch_losses

    # -- DDIM Inference (with optional physics guidance) -------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        coarse_np: np.ndarray,     # 2-D (H_coarse, W_coarse) in any unit
        ddim_steps: int = None,
        physics_guidance: bool = True,
        physics_weight: float = None,
        target_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Run DDIM super-resolution inference.

        coarse_np   : 2-D numpy array of the 12 km coarse field
        physics_guidance : if True, gradient-based physics correction is applied
                           at each denoising step (Step 2 of the roadmap)
        Returns     : 2-D numpy array of the 5 km super-resolved field
        """
        ddim_steps    = ddim_steps    or self.DDIM_S
        physics_weight = physics_weight or self.PHYS_W

        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for inference.")

        self.net.eval()

        # -- Normalise coarse input ------------------------------------------
        c_min, c_max = float(coarse_np.min()), float(coarse_np.max())
        coarse_n = 2 * (coarse_np - c_min) / (c_max - c_min + 1e-8) - 1

        # -- Target HR size -------------------------------------------------
        if target_size is None:
            H_out = int(round(coarse_np.shape[0] * self.SCALE))
            W_out = int(round(coarse_np.shape[1] * self.SCALE))
            target_size = (H_out, W_out)

        coarse_t = torch.tensor(coarse_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        cond     = F.interpolate(coarse_t, size=target_size, mode="bilinear", align_corners=False)

        # -- DDIM step indices (evenly spaced) ------------------------------
        step_indices = np.linspace(self.T - 1, 0, ddim_steps, dtype=int)

        # Start from pure noise
        x = torch.randn(1, 1, *target_size, device=self.device)

        for i, t_val in enumerate(step_indices):
            t_tensor = torch.tensor([t_val], device=self.device)

            # -- Predict noise with physics guidance (Step 2) --------------
            if physics_guidance and i > 0:
                # Enable grad just for guidance computation
                x_grad = x.detach().requires_grad_(True)
                with torch.enable_grad():
                    pred_n_g = self.net(x_grad, t_tensor, cond)
                    # Estimate x0 from current x_t
                    at       = self.alphas_cumprod[t_val].to(self.device)
                    x0_est   = (x_grad - math.sqrt(1 - at) * pred_n_g) / (math.sqrt(at) + 1e-8)
                    phys_pen = _mass_divergence_loss(x0_est)
                    phys_pen.backward()
                    # Physics gradient correction on x
                    guidance_grad = x_grad.grad.detach()

                with torch.no_grad():
                    pred_n = self.net(x, t_tensor, cond)
                    # Apply physics gradient step (gradient descent on divergence)
                    pred_n = pred_n + physics_weight * guidance_grad
            else:
                pred_n = self.net(x, t_tensor, cond)

            # -- DDIM update rule -------------------------------------------
            at  = self.alphas_cumprod[t_val].to(self.device)
            if i < len(step_indices) - 1:
                at_prev = self.alphas_cumprod[step_indices[i + 1]].to(self.device)
            else:
                at_prev = torch.tensor(1.0, device=self.device)

            x0_pred = (x - math.sqrt(1 - at) * pred_n) / (math.sqrt(at) + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)   # clip to valid range
            # Deterministic DDIM step (η = 0)
            x = math.sqrt(at_prev) * x0_pred + math.sqrt(1 - at_prev) * pred_n

        # -- Denormalise back to original units -----------------------------
        hr_norm = x.squeeze().cpu().numpy()
        hr_field = (hr_norm + 1) / 2 * (c_max - c_min) + c_min
        return hr_field.astype(np.float32)

    # -- Persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":      self.net.state_dict(),
            "T":                self.T,
            "DDIM_S":           self.DDIM_S,
            "SCALE":            self.SCALE,
            "PHYS_W":           self.PHYS_W,
            "betas":            self.betas,
            "alphas_cumprod":   self.alphas_cumprod,
            "trained":          self._trained,
        }, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[PGDM] Model saved -> {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MeteoraDiffusionModel":
        """Load a previously saved model from disk."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required.")
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        model  = cls(device=device)
        model.net.load_state_dict(ckpt["model_state"])
        model.T               = ckpt.get("T",             model.T)
        model.DDIM_S          = ckpt.get("DDIM_S",        model.DDIM_S)
        model.SCALE           = ckpt.get("SCALE",         model.SCALE)
        model.PHYS_W          = ckpt.get("PHYS_W",        model.PHYS_W)
        model.betas           = ckpt.get("betas",         model.betas)
        model.alphas_cumprod  = ckpt.get("alphas_cumprod",model.alphas_cumprod)
        model._trained        = ckpt.get("trained",       False)
        model.sqrt_alphas_cumprod   = torch.sqrt(model.alphas_cumprod)
        model.sqrt_one_minus_alphas = torch.sqrt(1.0 - model.alphas_cumprod)
        model.net.eval()
        print(f"[PGDM] Model loaded ← {path}  (trained={model._trained})")
        return model

    @property
    def is_trained(self) -> bool:
        return self._trained

    def parameter_count(self) -> int:
        if not _TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.net.parameters())
