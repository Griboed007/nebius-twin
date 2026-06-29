"""Anomaly scoring via prediction-vs-actual displacement.

Formula (bounded by construction, never clips):
    displacement = mean Euclidean distance in (lat, lon) degrees over all steps
    score = displacement / (displacement + scale)

where `scale` controls the inflection point.  At scale=ANOMALY_SCALE a
displacement equal to scale maps to 0.5 (matches ANOMALY_THRESHOLD semantics).

This is a monotone mapping R+ → [0,1) that:
  - Returns exactly 0.0 when predicted == actual (zero error)
  - Approaches 1.0 asymptotically for very large errors
  - Requires NO clamping — pydantic ge=0, le=1 always satisfied
"""

from __future__ import annotations

import math
from typing import List

from contracts.schema import TrajectoryPoint

# Scale parameter: displacement (in deg) that maps to anomaly_score ≈ 0.5.
# 0.01 degrees ≈ 1.1 km — a reasonable "suspicious jump" threshold.
ANOMALY_SCALE: float = 0.01


def score_anomaly(
    predicted: List[TrajectoryPoint],
    actual: List[TrajectoryPoint],
    scale: float = ANOMALY_SCALE,
) -> float:
    """Return a value in [0, 1] measuring how anomalous the prediction error is.

    Args:
        predicted: List of predicted TrajectoryPoints.
        actual:    List of actually-observed TrajectoryPoints (same length).
        scale:     Displacement (degrees) at which score = 0.5.

    Returns:
        Float in [0.0, 1.0].  0 = perfect prediction, →1 = large error.
    """
    if not predicted or not actual:
        return 0.0

    n = min(len(predicted), len(actual))
    total_sq = 0.0
    for p, a in zip(predicted[:n], actual[:n]):
        dlat = p.lat - a.lat
        dlon = p.lon - a.lon
        total_sq += dlat * dlat + dlon * dlon

    mean_displacement = math.sqrt(total_sq / n)
    # Bounded map: d / (d + s) ∈ [0, 1)
    score = mean_displacement / (mean_displacement + scale)
    # Clamp defensively (should never trigger, but protects against fp edge cases)
    return max(0.0, min(1.0, score))
