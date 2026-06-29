"""Frozen data contract for the semantic digital-twin pipeline.

SOURCE OF TRUTH for record shapes. Every module imports from here.
Implementers MUST NOT edit this file inside a worktree. Interface changes go
through an OpenSpec proposal that updates contracts/ on main, then rebase.

All times are UNIX seconds (float). All coordinates are WGS84 decimal degrees.
Velocities are in degrees/second in the (lat, lon) frame to keep the baseline
predictor unit-consistent with positions; convert to m/s only for reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Canonical constants (defaults; a caller may override explicitly) ---
HISTORY_LEN: int = 12          # observations fed to predict()
HORIZON: int = 6               # future steps predict() returns
ANOMALY_THRESHOLD: float = 0.5  # is_anomaly := score >= threshold
SAMPLE_PERIOD_S: float = 1.0    # nominal seconds between observations

# Canonical parquet column order for trajectory tables on object storage.
TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "agent_id", "t", "lat", "lon", "vlat", "vlon",
)
ENRICHED_EXTRA_COLUMNS: tuple[str, ...] = (
    "pred_lat", "pred_lon", "anomaly_score", "is_anomaly", "kind",
)


class TrajectoryPoint(BaseModel):
    """A single observed or predicted state of one agent at one instant."""
    t: float = Field(..., description="UNIX seconds")
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    vlat: float = 0.0
    vlon: float = 0.0


class Trajectory(BaseModel):
    """An ordered track for one agent. Points MUST be sorted by t ascending."""
    agent_id: str
    points: list[TrajectoryPoint]


class Prediction(BaseModel):
    """Model output for one agent: future points plus an anomaly judgement."""
    agent_id: str
    horizon: int = HORIZON
    points: list[TrajectoryPoint]
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool


class EnrichedPoint(TrajectoryPoint):
    """An observed point annotated with the per-point anomaly score."""
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool


class EnrichedTrajectory(BaseModel):
    """One agent's observed track plus its prediction and per-point anomalies.

    This is the unit the batch job writes and the twin layer ingests.
    Provenance invariant (enforced by twin/ and checked in verification):
    every predicted point references this agent_id; no enriched point exists
    without a parent agent in the same artifact.
    """
    agent_id: str
    observed: list[EnrichedPoint]
    prediction: Prediction
