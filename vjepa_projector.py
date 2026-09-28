"""
===============================================================================
AEGIS: V-JEPA 2 PARAMETER PROJECTION HEAD (GEOMORPHIC PRIOR PARAMETERIZATION)
===============================================================================
Projects frozen Meta V-JEPA 2 ViT-L latent embeddings into 2D hydrodynamic
parameter grids for Julia 2D Cellular Automata simulation using geomorphic
prior parameterization.

NOTE ON CALIBRATION / PROJECTION:
This module projects V-JEPA 2 latent representations into hydrodynamic
parameters (soil saturation and Manning's n roughness) by parameterizing
weights against synthetic coastal geomorphic spatial priors. It is an
engineered representation-to-parameter mapping, not a scientifically
calibrated empirical model fitted to paired ground-truth physical soil moisture data.

Architecture & Mechanics:
1. Ingests frozen Meta V-JEPA 2 ViT-L latent embeddings:
   - Shape: [Batch, 1568, 1024]
   - Spatial-temporal decomposition: 1568 = 8 temporal frames x (14x14 spatial patches)
2. Temporally pools features to [Batch, 1024, 14, 14] spatial feature maps.
3. Multi-layer convolutional projection trunk with bilinear spatial upsampling
   to project latent feature tokens to target hydrodynamic grid dimensions (e.g. 100x100).
4. Dual bounded output heads:
   - Soil Saturation Head: Sigmoid activation strictly bounded in [0.0, 1.0]
   - Manning's Roughness 'n' Head: Affine Sigmoid strictly bounded in [0.01, 0.15]
5. Grid exporters:
   - JSON export for Julia Oxygen.jl / simulate_ca.jl microservice consumption
   - CSV export for GIS / hydrodynamics validation
===============================================================================
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "perception_cache", "vjepa_projector_calibrated.pth")


class ParameterProjectionHead(nn.Module):
    """
    PyTorch projection network translating frozen V-JEPA 2 latent embeddings
    into 2D spatially varying hydrodynamic parameter grids for Julia physics solvers
    based on geomorphic prior parameterization.
    """
    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 256,
        target_shape: Tuple[int, int] = (100, 100)
    ):
        super().__init__()
        self.target_shape = target_shape
        self.in_channels = in_channels

        # Spatial-temporal reduction & projection trunk
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Spatial refinement post-upsampling
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # Head 1: Soil Saturation (bounded strictly [0.0, 1.0])
        self.head_saturation = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Head 2: Manning's Surface Roughness 'n' (bounded strictly [0.01, 0.15])
        self.head_manning = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.act = nn.GELU()
        self._initialize_calibrated_biases()

    def _initialize_calibrated_biases(self):
        """
        Initializes weights with realistic coastal baseline priors:
        - Baseline saturation centered at ~0.55 (wet coastal terrain)
        - Baseline Manning's n centered at ~0.035 (standard coastal floodplain)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Calibrate output biases so default sigmoid values match physical baselines
        # Sigmoid(0.20) ≈ 0.55 (Saturation baseline)
        if self.head_saturation[-1].bias is not None:
            nn.init.constant_(self.head_saturation[-1].bias, 0.20)

        # Manning's formula: 0.01 + 0.14 * Sigmoid(z)
        # To get n ≈ 0.035 -> 0.14 * Sigmoid(z) = 0.025 -> Sigmoid(z) ≈ 0.178 -> z ≈ -1.53
        if self.head_manning[-1].bias is not None:
            nn.init.constant_(self.head_manning[-1].bias, -1.53)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            features: [Batch, 1568, 1024] or [1568, 1024] from V-JEPA 2 ViT-L
        Returns:
            saturation_grid: [Batch, 100, 100] in [0.0, 1.0]
            manning_grid:    [Batch, 100, 100] in [0.01, 0.15]
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, 1568, 1024]

        B, N, C = features.shape

        # Decompose spatio-temporal tokens: 1568 = 8 temporal frames x 14 height x 14 width
        if N == 1568:
            x = features.view(B, 8, 14, 14, C)
            # Pool across temporal dimension -> [B, 14, 14, C]
            x = x.mean(dim=1)
            # Permute to PyTorch CNN format: [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        elif N == 196:
            # Single-frame 14x14 spatial tokens
            x = features.view(B, 14, 14, C).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 1568 or 196 spatial-temporal tokens, got {N}")

        # Trunk projection
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        # Bilinear spatial interpolation to simulation grid dimensions
        x = F.interpolate(x, size=self.target_shape, mode='bilinear', align_corners=False)
        x = self.act(self.bn3(self.conv3(x)))

        # Output Head 1: Soil Saturation [0.0, 1.0]
        sat_raw = self.head_saturation(x)
        saturation_grid = torch.sigmoid(sat_raw).squeeze(1)

        # Output Head 2: Manning's Surface Roughness n [0.01, 0.15]
        man_raw = self.head_manning(x)
        manning_grid = (0.01 + 0.14 * torch.sigmoid(man_raw)).squeeze(1)

        return saturation_grid, manning_grid


# =============================================================================
# CALIBRATION & TRAINING UTILITIES
# =============================================================================
def calibrate_projector_weights(
    projector: ParameterProjectionHead,
    sample_features: torch.Tensor,
    epochs: int = 40,
    lr: float = 1e-3,
    verbose: bool = False
) -> ParameterProjectionHead:
    """
    Parameterizes the projection head weights so spatial features align with synthetic
    coastal geomorphology priors (wetland saturation gradient, sand dune roughness).
    """
    device = next(projector.parameters()).device
    features = sample_features.to(device)

    # Construct target coastal distribution for Puri littoral zone:
    # Rows near coast (row 0 to 25) have higher water presence/saturation (0.75-0.90)
    # Inland rows (row 25 to 99) transition to drier inland ground (0.35-0.55)
    # Littoral forest/dune zone (cols 40-75, rows 5-25) has high Manning n (0.050-0.080)
    # Estuarine channel (cols 10-30) has lower Manning n (0.022-0.030)
    target_sat = torch.zeros(1, 100, 100, device=device)
    target_man = torch.full((1, 100, 100), 0.035, device=device)

    for i in range(100):
        # Base inland gradient
        dist_factor = np.exp(-i / 35.0)
        target_sat[0, i, :] = 0.40 + 0.48 * dist_factor

    # Estuarine wetland zone (SW littoral zone / Chilika approach)
    target_sat[0, :30, :40] = torch.clamp(target_sat[0, :30, :40] + 0.15, 0.0, 0.95)
    target_man[0, :30, :40] = 0.024  # Smooth water channel

    # Vegetated coastal sand dune zone (Balukhand reserve corridor)
    target_man[0, 5:25, 45:80] = 0.065  # Dense casuarina/mangrove vegetation

    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    projector.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        pred_sat, pred_man = projector(features)
        loss = loss_fn(pred_sat, target_sat) + 20.0 * loss_fn(pred_man, target_man)
        loss.backward()
        optimizer.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"   [Calibration Epoch {ep+1}/{epochs}] Loss: {loss.item():.6f}")

    projector.eval()
    return projector


def get_or_create_calibrated_projector(
    sample_features: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> ParameterProjectionHead:
    """Loads saved calibrated weights or initializes and calibrates a new head."""
    projector = ParameterProjectionHead(in_channels=1024, target_shape=(100, 100)).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
            projector.load_state_dict(state_dict)
            projector.eval()
            return projector
        except Exception:
            pass

    # Calibrate on provided or synthesized features
    if sample_features is None:
        sample_features = torch.randn(1, 1568, 1024, device=device)

    projector = calibrate_projector_weights(projector, sample_features, epochs=30, verbose=False)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    try:
        torch.save(projector.state_dict(), CHECKPOINT_PATH)
    except Exception:
        pass

    projector.eval()
    return projector


# =============================================================================
# EXPORTERS & INTEROP UTILITIES
# =============================================================================
def export_grids_to_json(
    saturation_grid: np.ndarray,
    manning_grid: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Exports 2D parameter grids directly to JSON for Julia microservice ingestion."""
    payload = {
        "format": "AEGIS_VJEPA2_CALIBRATED_GRIDS",
        "grid_shape": list(saturation_grid.shape),
        "mean_saturation": float(round(float(saturation_grid.mean()), 4)),
        "mean_manning_n": float(round(float(manning_grid.mean()), 4)),
        "min_saturation": float(round(float(saturation_grid.min()), 4)),
        "max_saturation": float(round(float(saturation_grid.max()), 4)),
        "min_manning_n": float(round(float(manning_grid.min()), 4)),
        "max_manning_n": float(round(float(manning_grid.max()), 4)),
        "saturation_grid": saturation_grid.round(4).tolist(),
        "manning_grid": manning_grid.round(5).tolist()
    }
    if metadata:
        payload["metadata"] = metadata

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def export_grids_to_csv(grid: np.ndarray, output_path: str):
    """Exports a 2D numpy array as a CSV grid."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, grid, delimiter=",", fmt="%.5f")
    return output_path


# =============================================================================
# END-TO-END PIPELINE ENTRY POINT
# =============================================================================
def project_vjepa_to_physics(
    tile_tensor: Optional[torch.Tensor] = None,
    target_shape: Tuple[int, int] = (100, 100),
    device_str: str = "auto"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Executes V-JEPA 2 ViT-L inference and projects latent tokens through
    the ParameterProjectionHead into calibrated 2D physical parameter grids.

    Returns:
        saturation_grid: [100, 100] float numpy array in [0.0, 1.0]
        manning_grid:    [100, 100] float numpy array in [0.01, 0.15]
        telemetry:       Dictionary with timing, summary statistics, and bounds
    """
    import time
    t0 = time.time()

    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 1. Load V-JEPA 2 ViT-L Backbone
    from perception_stage import load_vjepa2_vit_large, get_sentinel_tile
    encoder, device, _ = load_vjepa2_vit_large(str(device))

    if tile_tensor is None:
        tile_tensor = get_sentinel_tile()

    inp = tile_tensor.half().to(device) if device.type == "cuda" else tile_tensor.to(device)

    # 2. Extract Latent Embeddings (Frozen ViT-L forward pass)
    with torch.no_grad():
        features = encoder(inp)  # [1, 1568, 1024]

    # 3. Project through Calibrated ParameterProjectionHead
    projector = get_or_create_calibrated_projector(features.float(), str(device))
    with torch.no_grad():
        sat_t, man_t = projector(features.float())

    sat_grid = sat_t.squeeze(0).cpu().numpy().astype(np.float64)
    man_grid = man_t.squeeze(0).cpu().numpy().astype(np.float64)

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    telemetry = {
        "status": "SUCCESS",
        "model": "V-JEPA 2 ViT-L + ParameterProjectionHead",
        "device": str(device),
        "target_grid_shape": list(target_shape),
        "saturation_bounds": [float(round(sat_grid.min(), 4)), float(round(sat_grid.max(), 4))],
        "saturation_mean": float(round(sat_grid.mean(), 4)),
        "manning_bounds": [float(round(man_grid.min(), 4)), float(round(man_grid.max(), 4))],
        "manning_mean": float(round(man_grid.mean(), 4)),
        "elapsed_ms": elapsed_ms
    }

    return sat_grid, man_grid, telemetry


if __name__ == "__main__":
    print("Testing ParameterProjectionHead...")
    sat, man, stats = project_vjepa_to_physics()
    print("Execution complete:")
    print(json.dumps(stats, indent=2))
    print(f"Saturation grid shape: {sat.shape}, Manning grid shape: {man.shape}")
