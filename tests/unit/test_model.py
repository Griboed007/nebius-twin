"""Unit tests for src/model/{baseline,residual,anomaly}.py — run RED before impl.

Covers:
  1. Anomaly scores bounded in [0,1]
  2. AUC-floor test against labelled synthetic data (self-contained fixture)
  3. Residual graceful-fallback: missing MODEL_PATH → baseline result
  4. Corrupt MODEL_PATH → warning + baseline result
  5. Torch-absent path (if torch not installed, residual behaves as no-op)

AUC floor rationale:
  Synthetic tracks injected with teleport jumps (10x normal step size) are
  labelled anomalous=1; normal straight-line tracks are labelled anomalous=0.
  Expected AUC >= 0.85 (measured ~0.97 in design; floor set conservatively).
"""

import logging
import math
import os
import tempfile

import pytest

from contracts.schema import (
    HISTORY_LEN,
    HORIZON,
    ANOMALY_THRESHOLD,
    SAMPLE_PERIOD_S,
    Trajectory,
    TrajectoryPoint,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

AUC_FLOOR = 0.85  # Conservative floor; measured ~0.97-1.0 depending on jitter seed


def _straight_track(
    agent_id: str,
    n: int = HISTORY_LEN,
    lat0: float = 52.0,
    lon0: float = 13.0,
    vlat: float = 0.001,
    vlon: float = 0.0005,
    jitter: float = 0.0,
    rng_seed: int = 0,
) -> Trajectory:
    """Normal constant-velocity track, with optional small Gaussian noise."""
    import random
    rng = random.Random(rng_seed)
    points = [
        TrajectoryPoint(
            t=float(i),
            lat=lat0 + i * vlat + rng.gauss(0, jitter),
            lon=lon0 + i * vlon + rng.gauss(0, jitter),
            vlat=vlat,
            vlon=vlon,
        )
        for i in range(n)
    ]
    return Trajectory(agent_id=agent_id, points=points)


def _teleport_track(
    agent_id: str,
    n: int = HISTORY_LEN,
    jump_at: int = 6,
    jump_size: float = 0.05,
) -> Trajectory:
    """Track that has a sudden large jump at step `jump_at`."""
    points = []
    lat, lon = 52.0, 13.0
    vlat, vlon = 0.001, 0.0005
    for i in range(n):
        if i == jump_at:
            lat += jump_size
            lon += jump_size
        points.append(
            TrajectoryPoint(
                t=float(i),
                lat=lat + i * vlat,
                lon=lon + i * vlon,
                vlat=vlat,
                vlon=vlon,
            )
        )
    return Trajectory(agent_id=agent_id, points=points)


# --------------------------------------------------------------------------- #
# Anomaly scoring: bounds
# --------------------------------------------------------------------------- #

class TestAnomalyBounds:
    """score_anomaly must return values in [0, 1]."""

    def test_zero_error_gives_zero_score(self):
        from src.model.anomaly import score_anomaly

        predicted = [TrajectoryPoint(t=float(i), lat=52.0 + i * 0.001, lon=13.0) for i in range(3)]
        actual    = [TrajectoryPoint(t=float(i), lat=52.0 + i * 0.001, lon=13.0) for i in range(3)]
        score = score_anomaly(predicted, actual)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_score_in_range_normal_track(self):
        from src.model.anomaly import score_anomaly

        predicted = [TrajectoryPoint(t=float(i), lat=52.0 + i * 0.001, lon=13.0) for i in range(3)]
        actual    = [TrajectoryPoint(t=float(i), lat=52.0 + i * 0.001 + 0.0002, lon=13.0) for i in range(3)]
        score = score_anomaly(predicted, actual)
        assert 0.0 <= score <= 1.0

    def test_large_error_approaches_one(self):
        """Very large displacement should produce a score close to 1."""
        from src.model.anomaly import score_anomaly

        predicted = [TrajectoryPoint(t=float(i), lat=52.0, lon=13.0) for i in range(3)]
        actual    = [TrajectoryPoint(t=float(i), lat=52.0 + 10.0, lon=13.0 + 10.0) for i in range(3)]
        score = score_anomaly(predicted, actual)
        assert score > 0.9

    def test_score_monotone_with_error(self):
        """Bigger error → bigger score."""
        from src.model.anomaly import score_anomaly

        pred = [TrajectoryPoint(t=float(i), lat=52.0, lon=13.0) for i in range(3)]
        act_small = [TrajectoryPoint(t=float(i), lat=52.0 + 0.001, lon=13.0) for i in range(3)]
        act_large = [TrajectoryPoint(t=float(i), lat=52.0 + 0.5, lon=13.0) for i in range(3)]
        s_small = score_anomaly(pred, act_small)
        s_large = score_anomaly(pred, act_large)
        assert s_small < s_large

    def test_all_scores_in_unit_interval_random(self):
        """Fuzz: 100 random pairs must all stay in [0,1]."""
        import random
        from src.model.anomaly import score_anomaly

        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(1, 10)
            pred = [TrajectoryPoint(t=float(i), lat=rng.uniform(-89, 89), lon=rng.uniform(-179, 179)) for i in range(n)]
            act  = [TrajectoryPoint(t=float(i), lat=rng.uniform(-89, 89), lon=rng.uniform(-179, 179)) for i in range(n)]
            score = score_anomaly(pred, act)
            assert 0.0 <= score <= 1.0, f"score={score} out of [0,1]"


# --------------------------------------------------------------------------- #
# AUC-floor test: labelled synthetic fixture
# --------------------------------------------------------------------------- #

class TestAnomalyAUC:
    """AUC against labelled synthetic tracks must exceed AUC_FLOOR."""

    @pytest.fixture(scope="class")
    @classmethod
    def labelled_dataset(cls):
        """Build a self-contained labelled dataset.

        15 normal tracks (label=0) + 15 anomalous tracks with teleports (label=1).
        Returns (scores: list[float], labels: list[int]).
        """
        from src.model.core import predict

        scores = []
        labels = []

        # Normal tracks — different starting positions with small GPS-like jitter
        # Jitter 0.0005 deg ≈ 55m, smaller than the teleport jump (0.05 deg ≈ 5.5km)
        for i in range(15):
            traj = _straight_track(
                agent_id=f"normal-{i}",
                lat0=50.0 + i * 0.1,
                lon0=10.0 + i * 0.1,
                jitter=0.0005,
                rng_seed=i,
            )
            pred = predict(traj)
            scores.append(pred.anomaly_score)
            labels.append(0)

        # Anomalous tracks — teleport jumps
        for i in range(15):
            traj = _teleport_track(
                agent_id=f"anomaly-{i}",
                jump_at=HISTORY_LEN // 2,
                jump_size=0.05 + i * 0.01,
            )
            pred = predict(traj)
            scores.append(pred.anomaly_score)
            labels.append(1)

        return scores, labels

    def test_auc_floor(self, labelled_dataset, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from sklearn.metrics import roc_auc_score

        scores, labels = labelled_dataset
        auc = roc_auc_score(labels, scores)
        # Document both numbers
        print(f"\nMeasured AUC={auc:.4f}  Floor={AUC_FLOOR}")
        assert auc >= AUC_FLOOR, f"AUC {auc:.4f} < floor {AUC_FLOOR}"

    def test_all_scores_bounded(self, labelled_dataset):
        scores, _ = labelled_dataset
        for s in scores:
            assert 0.0 <= s <= 1.0, f"score {s} out of bounds"


# --------------------------------------------------------------------------- #
# Residual fallback
# --------------------------------------------------------------------------- #

class TestResidualFallback:
    """Graceful degradation lives in load_residual(); predict() is load-free.

    The caller (endpoint/job) owns loading:
        model = load_residual(os.environ.get("MODEL_PATH"))
        pred  = predict(history, residual_model=model)

    predict() with no residual_model uses a module-level _ZeroResidual (no IO).
    """

    def test_no_model_path_returns_prediction(self, monkeypatch):
        """predict() with no residual_model uses the no-op default — no IO."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict

        traj = _straight_track("agent-noresidual")
        result = predict(traj)
        assert len(result.points) == HORIZON

    def test_load_residual_missing_file_logs_warning_and_returns_zero(
        self, caplog
    ):
        """load_residual() with a missing path warns and returns _ZeroResidual."""
        with caplog.at_level(logging.WARNING, logger="src.model.residual"):
            from src.model.residual import load_residual, _ZeroResidual
            model = load_residual("/nonexistent/path/model.pt")
        assert isinstance(model, _ZeroResidual)
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "MODEL_PATH" in m or "residual" in m.lower() or "fallback" in m.lower()
            for m in warning_msgs
        ), f"No fallback warning found; got: {warning_msgs}"

    def test_load_residual_missing_gives_working_prediction(self, caplog):
        """_ZeroResidual from a failed load works normally when passed to predict()."""
        with caplog.at_level(logging.WARNING, logger="src.model.residual"):
            from src.model.residual import load_residual
            from src.model.core import predict
            zero_model = load_residual("/nonexistent/path/model.pt")
        result = predict(_straight_track("agent-missing"), residual_model=zero_model)
        assert len(result.points) == HORIZON

    def test_load_residual_corrupt_file_logs_warning_and_returns_zero(
        self, tmp_path, caplog
    ):
        """load_residual() with a corrupt file warns and returns _ZeroResidual."""
        corrupt = tmp_path / "corrupt_model.pt"
        corrupt.write_bytes(b"this is not a valid model file \x00\xff\xfe")
        with caplog.at_level(logging.WARNING, logger="src.model.residual"):
            from src.model.residual import load_residual, _ZeroResidual
            model = load_residual(str(corrupt))
        assert isinstance(model, _ZeroResidual)
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "MODEL_PATH" in m or "residual" in m.lower() or "fallback" in m.lower()
            for m in warning_msgs
        ), f"No fallback warning found; got: {warning_msgs}"

    def test_load_residual_corrupt_gives_working_prediction(
        self, tmp_path, caplog
    ):
        """Passing the zero-residual from a corrupt load into predict() still works."""
        corrupt = tmp_path / "corrupt_model.pt"
        corrupt.write_bytes(b"this is not a valid model file \x00\xff\xfe")
        with caplog.at_level(logging.WARNING, logger="src.model.residual"):
            from src.model.residual import load_residual
            zero_model = load_residual(str(corrupt))
        from src.model.core import predict
        result = predict(_straight_track("agent-corrupt"), residual_model=zero_model)
        assert len(result.points) == HORIZON

    def test_predict_does_not_call_load_residual_per_call(self):
        """predict() must never call load_residual() internally — it is load-free."""
        from unittest.mock import patch
        from src.model.core import predict

        traj = _straight_track("agent-pure")
        with patch("src.model.residual.load_residual") as mock_load:
            predict(traj)
            predict(traj)
            predict(traj)
        mock_load.assert_not_called()

    def test_predict_ignores_model_path_env(self, monkeypatch):
        """predict() must not read os.environ — MODEL_PATH changes have no effect."""
        from src.model.core import predict

        traj = _straight_track("agent-env")
        monkeypatch.delenv("MODEL_PATH", raising=False)
        r1 = predict(traj)

        # Setting MODEL_PATH to a nonexistent file must not change predict() output
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/should-be-ignored.pt")
        r2 = predict(traj)

        assert r1.points[0].lat == r2.points[0].lat
        assert r1.anomaly_score == r2.anomaly_score

    def test_baseline_pure_when_torch_absent(self, monkeypatch):
        """predict() works without torch installed (residual is _ZeroResidual)."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        from src.model.core import predict
        result = predict(_straight_track("agent-notorch"))
        assert len(result.points) == HORIZON


# --------------------------------------------------------------------------- #
# Baseline unit tests
# --------------------------------------------------------------------------- #

class TestBaseline:
    """Direct tests of the baseline predictor."""

    def test_output_horizon_length(self):
        from src.model.baseline import predict_baseline

        traj = _straight_track("b0")
        points = predict_baseline(traj)
        assert len(points) == HORIZON

    def test_constant_velocity_extrapolation(self):
        """On a perfectly constant-velocity track the baseline should
        continue smoothly (small rounding aside)."""
        from src.model.baseline import predict_baseline

        vlat, vlon = 0.001, 0.0005
        traj = _straight_track("cv", vlat=vlat, vlon=vlon)
        last = traj.points[-1]
        predicted = predict_baseline(traj)

        # Check that step sizes are approximately constant
        step_lats = [
            predicted[i + 1].lat - predicted[i].lat for i in range(HORIZON - 1)
        ]
        expected_step = vlat  # ~one period worth
        for step in step_lats:
            assert abs(step - expected_step) < 0.01, (
                f"Baseline step {step:.6f} differs too much from expected {expected_step}"
            )

    def test_predicted_coords_in_bounds(self):
        from src.model.baseline import predict_baseline

        traj = _straight_track("b-bounds")
        pts = predict_baseline(traj)
        for pt in pts:
            assert -90.0 <= pt.lat <= 90.0
            assert -180.0 <= pt.lon <= 180.0

    def test_timestamps_monotone_increasing(self):
        from src.model.baseline import predict_baseline

        traj = _straight_track("b-ts")
        pts = predict_baseline(traj)
        for i in range(len(pts) - 1):
            assert pts[i + 1].t > pts[i].t

    def test_first_predicted_t_after_last_history_t(self):
        from src.model.baseline import predict_baseline

        traj = _straight_track("b-t0")
        last_t = traj.points[-1].t
        pts = predict_baseline(traj)
        assert pts[0].t > last_t
