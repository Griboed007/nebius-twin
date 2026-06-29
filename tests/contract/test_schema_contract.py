"""Contract smoke test for the frozen data schema (proposal 000)."""
import pytest
from pydantic import ValidationError
from contracts.schema import TrajectoryPoint, EnrichedTrajectory, HORIZON


def test_valid_point():
    p = TrajectoryPoint(t=0.0, lat=52.2391, lon=6.8570, vlat=0.0, vlon=0.0)
    assert -90 <= p.lat <= 90


def test_invalid_lat_rejected():
    with pytest.raises(ValidationError):
        TrajectoryPoint(t=0.0, lat=200.0, lon=0.0)


def test_horizon_constant_positive():
    assert HORIZON > 0
