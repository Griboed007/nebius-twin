"""Contract tests for src/model/core.py — run RED before implementation.

These tests verify the public contract of predict():
  - Returns a Prediction with len(points) == HORIZON
  - Is deterministic (same input → same output)
  - Works with MODEL_PATH unset (CPU, no trained artifact)
  - Output satisfies pydantic schema constraints
"""

import os
import sys

import pytest

from contracts.schema import (
    HISTORY_LEN,
    HORIZON,
    ANOMALY_THRESHOLD,
    Trajectory,
    TrajectoryPoint,
    Prediction,
)


def _make_trajectory(agent_id: str = "agent-0", n: int = HISTORY_LEN) -> Trajectory:
    """Straight-line track moving north at 0.001 deg/s."""
    points = [
        TrajectoryPoint(
            t=float(i),
            lat=52.0 + i * 0.001,
            lon=13.0 + i * 0.0005,
            vlat=0.001,
            vlon=0.0005,
        )
        for i in range(n)
    ]
    return Trajectory(agent_id=agent_id, points=points)


class TestPredictContractShape:
    """predict() must return a Prediction with exactly HORIZON points."""

    def test_returns_prediction_instance(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        assert isinstance(result, Prediction)

    def test_horizon_length(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        assert len(result.points) == HORIZON

    def test_agent_id_propagated(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        traj = _make_trajectory(agent_id="vehicle-42")
        result = predict(traj)
        assert result.agent_id == "vehicle-42"

    def test_horizon_field_matches_constant(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        assert result.horizon == HORIZON

    def test_anomaly_score_bounded(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_is_anomaly_consistent_with_score(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        expected_is_anomaly = result.anomaly_score >= ANOMALY_THRESHOLD
        assert result.is_anomaly == expected_is_anomaly

    def test_predicted_coords_within_bounds(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        for pt in result.points:
            assert -90.0 <= pt.lat <= 90.0
            assert -180.0 <= pt.lon <= 180.0


class TestPredictDeterminism:
    """Same input → same output, no global state."""

    def test_deterministic_repeated_calls(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        traj = _make_trajectory()
        result1 = predict(traj)
        result2 = predict(traj)
        for p1, p2 in zip(result1.points, result2.points):
            assert p1.t == p2.t
            assert p1.lat == p2.lat
            assert p1.lon == p2.lon
        assert result1.anomaly_score == result2.anomaly_score

    def test_different_inputs_give_different_outputs(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        traj_a = _make_trajectory(agent_id="a")
        traj_b = Trajectory(
            agent_id="b",
            points=[
                TrajectoryPoint(
                    t=float(i),
                    lat=40.0 + i * 0.002,
                    lon=-74.0 + i * 0.001,
                )
                for i in range(HISTORY_LEN)
            ],
        )
        r_a = predict(traj_a)
        r_b = predict(traj_b)
        # They start from different positions so first predicted point differs
        assert r_a.points[0].lat != r_b.points[0].lat

    def test_no_model_path_cpu_only(self, monkeypatch):
        """Must succeed on CPU with no MODEL_PATH set."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        result = predict(_make_trajectory())
        assert len(result.points) == HORIZON


class TestPredictImportSafety:
    """Importing core must not cause side-effects or fail without MODEL_PATH."""

    def test_import_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        # The import itself should be side-effect-free
        import importlib
        import src.model.core as core_mod
        importlib.reload(core_mod)  # reload to re-test import path
        assert hasattr(core_mod, "predict")
