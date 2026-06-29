"""Optional residual corrector for the baseline predictor.

Behavior matrix:
  MODEL_PATH unset          → no-op (zero correction), no warning
  MODEL_PATH set, file OK   → load and apply the tiny net
  MODEL_PATH set, file bad  → log WARNING, fall back to zero correction
  torch not installed       → always zero correction regardless of MODEL_PATH

The residual is intentionally a small optional improvement.  The baseline is
always the safety net.  This module MUST NOT be imported inside predict(); it
is loaded once by the caller and passed in, keeping predict() pure.

Design note:
  To keep predict() import-safe and side-effect-free this module exposes:
    load_residual(path: str | None) -> ResidualModel | None
    apply_residual(residual_model, baseline_points) -> list[TrajectoryPoint]

  The caller (core.py) calls load_residual() once per predict() invocation
  (or pre-loads and caches outside predict()); apply_residual() is pure given
  a loaded model.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, TYPE_CHECKING

from contracts.schema import HORIZON, TrajectoryPoint

logger = logging.getLogger(__name__)

# Try to import torch; if absent the residual is permanently a no-op.
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


class _ZeroResidual:
    """Placeholder that always returns zero correction — no torch needed."""

    def apply(self, baseline_points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        return baseline_points


if _TORCH_AVAILABLE:
    class _TinyResidualNet(nn.Module):  # type: ignore[misc]
        """Small MLP: takes last 2 features (vlat, vlon) → correction (dlat, dlon)."""

        INPUT_DIM = 2   # (vlat, vlon) of last observed step
        HIDDEN    = 16
        OUTPUT_DIM = 2  # (dlat_correction, dlon_correction) per step

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(self.INPUT_DIM, self.HIDDEN),
                nn.Tanh(),
                nn.Linear(self.HIDDEN, self.OUTPUT_DIM * HORIZON),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.net(x)

    class _TorchResidual:
        """Loaded torch residual model."""

        def __init__(self, net: "_TinyResidualNet") -> None:
            self._net = net
            self._net.eval()

        def apply(self, baseline_points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
            """Apply correction to baseline points; returns corrected list."""
            if not baseline_points:
                return baseline_points
            # Feature: use velocity of first predicted point (proxy for last observed)
            vlat = baseline_points[0].vlat
            vlon = baseline_points[0].vlon
            x = torch.tensor([[vlat, vlon]], dtype=torch.float32)
            with torch.no_grad():
                delta = self._net(x).squeeze(0)  # shape: (HORIZON*2,)
            result = []
            for i, pt in enumerate(baseline_points):
                dlat = float(delta[2 * i])
                dlon = float(delta[2 * i + 1])
                from src.model.baseline import _clamp_lat, _clamp_lon
                result.append(
                    TrajectoryPoint(
                        t=pt.t,
                        lat=_clamp_lat(pt.lat + dlat),
                        lon=_clamp_lon(pt.lon + dlon),
                        vlat=pt.vlat,
                        vlon=pt.vlon,
                    )
                )
            return result


def load_residual(model_path: Optional[str]) -> _ZeroResidual:
    """Load the residual model from `model_path`.

    Returns a zero-correction model (no-op) if:
      - model_path is None or empty
      - torch is not installed
      - the file is missing or corrupt

    Never raises; logs a WARNING on any failure.
    """
    if not model_path:
        return _ZeroResidual()

    if not _TORCH_AVAILABLE:
        logger.warning(
            "MODEL_PATH=%s set but torch is not installed; "
            "residual fallback to baseline (zero correction).",
            model_path,
        )
        return _ZeroResidual()

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MODEL_PATH file not found: {model_path}")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        net = _TinyResidualNet()
        net.load_state_dict(state)
        logger.info("Loaded residual model from %s", model_path)
        return _TorchResidual(net)  # type: ignore[return-value]
    except Exception as exc:
        logger.warning(
            "Failed to load residual from MODEL_PATH=%s (%s); "
            "residual fallback to baseline (zero correction).",
            model_path,
            exc,
        )
        return _ZeroResidual()


def apply_residual(
    residual_model: _ZeroResidual,
    baseline_points: List[TrajectoryPoint],
) -> List[TrajectoryPoint]:
    """Apply residual correction to baseline_points. Pure given loaded model."""
    return residual_model.apply(baseline_points)
