"""Constant-velocity baseline predictor.

Pure, CPU-only, no trained artifact required.

Algorithm:
  1. Compute finite-difference velocity from the last HISTORY_LEN points using
     actual time deltas (falls back to SAMPLE_PERIOD_S if dt == 0).
  2. Average the velocity over a short trailing window (min(3, HISTORY_LEN-1)
     steps) for robustness against single-step noise.
  3. Extrapolate forward HORIZON steps using that average velocity.

Predicted coordinates are clamped to valid WGS84 ranges to satisfy pydantic
constraints even for large extrapolations.
"""

from __future__ import annotations

import math
from typing import List

from contracts.schema import (
    HISTORY_LEN,
    HORIZON,
    SAMPLE_PERIOD_S,
    Trajectory,
    TrajectoryPoint,
)

_VELOCITY_WINDOW = 3  # number of finite-difference steps to average


def predict_baseline(history: Trajectory) -> List[TrajectoryPoint]:
    """Return HORIZON predicted TrajectoryPoints from `history`.

    Args:
        history: A Trajectory with at least 2 points (ideally HISTORY_LEN).

    Returns:
        List of HORIZON TrajectoryPoint objects, timestamps strictly increasing
        from the last observed point.
    """
    points = history.points
    if len(points) < 2:
        # Degenerate: no motion, project stationary
        last = points[-1]
        dt = SAMPLE_PERIOD_S
        return [
            TrajectoryPoint(
                t=last.t + (i + 1) * dt,
                lat=_clamp_lat(last.lat),
                lon=_clamp_lon(last.lon),
                vlat=0.0,
                vlon=0.0,
            )
            for i in range(HORIZON)
        ]

    # Compute finite-difference velocities over last _VELOCITY_WINDOW steps
    window = min(_VELOCITY_WINDOW, len(points) - 1)
    vlat_acc, vlon_acc = 0.0, 0.0
    dt_last = SAMPLE_PERIOD_S  # used for timestamp stepping

    for i in range(1, window + 1):
        p_prev = points[-i - 1]
        p_curr = points[-i]
        dt = p_curr.t - p_prev.t
        if dt <= 0:
            dt = SAMPLE_PERIOD_S
        vlat_acc += (p_curr.lat - p_prev.lat) / dt
        vlon_acc += (p_curr.lon - p_prev.lon) / dt
        if i == 1:
            dt_last = dt  # use the most recent interval for future spacing

    avg_vlat = vlat_acc / window
    avg_vlon = vlon_acc / window

    last = points[-1]
    predicted: List[TrajectoryPoint] = []
    for i in range(HORIZON):
        step = i + 1
        new_lat = last.lat + avg_vlat * dt_last * step
        new_lon = last.lon + avg_vlon * dt_last * step
        predicted.append(
            TrajectoryPoint(
                t=last.t + dt_last * step,
                lat=_clamp_lat(new_lat),
                lon=_clamp_lon(new_lon),
                vlat=avg_vlat,
                vlon=avg_vlon,
            )
        )
    return predicted


def _clamp_lat(lat: float) -> float:
    return max(-90.0, min(90.0, lat))


def _clamp_lon(lon: float) -> float:
    return max(-180.0, min(180.0, lon))
