"""Public predict() contract for the model module.

predict(history: Trajectory) -> Prediction

Purity guarantee:
  - No global mutable state; no module-level file IO.
  - No network calls.
  - Deterministic: same history + same env → same output.
  - Import-safe: importing this module performs no IO.
  - No os.environ read and no file/torch load inside predict().

How anomaly score is computed (retrodiction):
  The last `RETRO_STEPS` of `history` are withheld; the baseline predicts them
  from the preceding observations; score_anomaly compares prediction to actuals.
  This reuses the predictor for anomaly detection with no extra model.

Residual loading — caller responsibility, NOT predict()'s:
  predict() accepts an optional `residual_model` param.  The caller (endpoint
  or batch job) is responsible for calling `residual.load_residual(path)` ONCE
  at startup and passing the result in.  When `residual_model` is None (the
  default), predict() uses a module-level no-op `_ZeroResidual` — zero file IO.

  Example endpoint/job usage:
      from src.model import core
      from src.model.residual import load_residual
      import os

      residual = load_residual(os.environ.get("MODEL_PATH"))  # once at startup

      for traj in stream:
          pred = core.predict(traj, residual_model=residual)  # pure, no IO
"""

from __future__ import annotations

from contracts.schema import (
    ANOMALY_THRESHOLD,
    HISTORY_LEN,
    HORIZON,
    Prediction,
    Trajectory,
    TrajectoryPoint,
)
from src.model.anomaly import score_anomaly
from src.model.baseline import predict_baseline
from src.model.residual import _ZeroResidual, apply_residual

# Number of tail steps used for retrodiction anomaly scoring.
# Must be < HISTORY_LEN so there are enough prefix steps to predict from.
RETRO_STEPS: int = min(3, HISTORY_LEN - 2)

# Module-level no-op residual — used when caller does not provide one.
# Created once at import time; stateless, no file IO.
_DEFAULT_RESIDUAL = _ZeroResidual()


def predict(
    history: Trajectory,
    residual_model: _ZeroResidual | None = None,
) -> Prediction:
    """Return HORIZON predicted points and an anomaly score for `history`.

    Pure: no os.environ read, no file/torch load, no network call inside this
    function.  Deterministic given (history, residual_model).

    Args:
        history:        Trajectory with at least 2 points (ideally HISTORY_LEN).
        residual_model: A pre-loaded residual (from residual.load_residual()).
                        If None, the module-level no-op (_ZeroResidual) is used.
                        Load once at startup; do NOT pass None and expect
                        predict() to read MODEL_PATH — it never does.

    Returns:
        Prediction satisfying the contracts/schema.py contract.
    """
    if residual_model is None:
        residual_model = _DEFAULT_RESIDUAL

    # --- Anomaly score via retrodiction (no future labels needed) ---
    anomaly_score = _retrodiction_score(history)

    # --- Forward prediction: baseline + optional residual correction ---
    baseline_points = predict_baseline(history)
    final_points = apply_residual(residual_model, baseline_points)

    return Prediction(
        agent_id=history.agent_id,
        horizon=HORIZON,
        points=final_points,
        anomaly_score=anomaly_score,
        is_anomaly=anomaly_score >= ANOMALY_THRESHOLD,
    )


def _retrodiction_score(history: Trajectory) -> float:
    """Score history by predicting its tail from its prefix, comparing to actuals.

    Uses RETRO_STEPS tail points as "held-out" ground truth.
    If history is too short to split, returns 0.0.
    """
    points = history.points
    split = len(points) - RETRO_STEPS
    if split < 2:
        return 0.0

    prefix_traj = Trajectory(
        agent_id=history.agent_id,
        points=points[:split],
    )
    predicted = predict_baseline(prefix_traj)[:RETRO_STEPS]
    actual = points[split : split + RETRO_STEPS]

    return score_anomaly(predicted, actual)
