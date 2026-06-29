"""Tests for src/data/synthetic.py and src/data/io.py.

Written BEFORE implementation (red → green TDD).

Label column contract for downstream consumers (020 model eval):
  - ``is_anomaly_gt`` (bool)  — ground-truth anomaly flag per point
  - ``kind`` (str)            — one of {"normal","sudden_stop","gps_jump","geofence_exit"}
  Join key: (agent_id, t).  These differ from the *predicted* ``is_anomaly``
  in ENRICHED_EXTRA_COLUMNS so there is no name collision when joining.
"""

import hashlib
import tempfile
import pathlib

import pandas as pd
import pytest

from contracts.schema import (
    TRAJECTORY_COLUMNS,
    TrajectoryPoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ANOMALY_KINDS = {"sudden_stop", "gps_jump", "geofence_exit"}

# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    def test_byte_identical_files(self, tmp_path):
        """Two runs with the same seed and params must produce byte-identical parquet."""
        from src.data.synthetic import generate
        from src.data.io import write_canonical

        df1 = generate(n_agents=3, n_steps=50, seed=42)
        df2 = generate(n_agents=3, n_steps=50, seed=42)

        p1 = tmp_path / "a.parquet"
        p2 = tmp_path / "b.parquet"
        write_canonical(df1, p1)
        write_canonical(df2, p2)

        assert _sha256_file(p1) == _sha256_file(p2), (
            "Identical seed/params must yield byte-identical parquet files"
        )

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        from src.data.synthetic import generate

        df1 = generate(n_agents=3, n_steps=50, seed=1)
        df2 = generate(n_agents=3, n_steps=50, seed=2)

        # Lat values should differ somewhere
        assert not df1["lat"].equals(df2["lat"])


# ---------------------------------------------------------------------------
# Schema validation — every row against TrajectoryPoint
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_every_row_validates(self):
        """Every row must validate against TrajectoryPoint (the 5-field subset)."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=30, seed=7)

        for _, row in df.iterrows():
            # agent_id is NOT in TrajectoryPoint — validate the 5 relevant fields
            point = TrajectoryPoint(
                t=row["t"],
                lat=row["lat"],
                lon=row["lon"],
                vlat=row["vlat"],
                vlon=row["vlon"],
            )
            # pydantic v2 — construction validates; no extra assertion needed
            assert isinstance(point, TrajectoryPoint)

    def test_canonical_columns_present(self):
        """DataFrame must include all TRAJECTORY_COLUMNS."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=30, seed=7)
        for col in TRAJECTORY_COLUMNS:
            assert col in df.columns, f"Missing canonical column: {col}"

    def test_label_columns_present(self):
        """DataFrame must include ground-truth label columns."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=30, seed=7)
        assert "is_anomaly_gt" in df.columns
        assert "kind" in df.columns

    def test_label_column_dtype(self):
        """is_anomaly_gt must be bool; kind must be string (object/StringDtype)."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=30, seed=7)
        assert df["is_anomaly_gt"].dtype == bool
        assert df["kind"].dtype == object or pd.api.types.is_string_dtype(df["kind"])

    def test_kind_values_valid(self):
        """Every kind value must be one of the four allowed strings."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=60, seed=7)
        allowed = {"normal"} | ANOMALY_KINDS
        unexpected = set(df["kind"].unique()) - allowed
        assert not unexpected, f"Unexpected kind values: {unexpected}"

    def test_agent_id_is_string(self):
        """agent_id must be a string column."""
        from src.data.synthetic import generate

        df = generate(n_agents=2, n_steps=10, seed=7)
        assert pd.api.types.is_string_dtype(df["agent_id"]) or df["agent_id"].dtype == object

    def test_lat_lon_in_bounds(self):
        """All lat/lon values must satisfy WGS84 bounds (including anomalous rows)."""
        from src.data.synthetic import generate

        df = generate(n_agents=5, n_steps=100, seed=99)
        assert df["lat"].between(-90.0, 90.0).all(), "lat out of [-90, 90]"
        assert df["lon"].between(-180.0, 180.0).all(), "lon out of [-180, 180]"


# ---------------------------------------------------------------------------
# Label presence — anomaly types and fraction
# ---------------------------------------------------------------------------

class TestLabelPresence:
    # Use a larger dataset so the anomaly_rate constraint is comfortable to satisfy
    N_AGENTS = 5
    N_STEPS = 200
    SEED = 42
    ANOMALY_RATE = 0.05
    TOLERANCE = 0.04  # ± 4 percentage points

    @pytest.fixture(scope="class")
    @classmethod
    def labelled_df(cls):
        from src.data.synthetic import generate
        return generate(
            n_agents=cls.N_AGENTS,
            n_steps=cls.N_STEPS,
            seed=cls.SEED,
            anomaly_rate=cls.ANOMALY_RATE,
        )

    def test_at_least_one_of_each_anomaly_type(self, labelled_df):
        """Each of the three anomaly types must appear at least once."""
        present = set(labelled_df.loc[labelled_df["is_anomaly_gt"], "kind"].unique())
        for kind in ANOMALY_KINDS:
            assert kind in present, f"Anomaly type '{kind}' has zero labelled rows"

    def test_labelled_fraction_within_tolerance(self, labelled_df):
        """Fraction of anomalous rows must be within TOLERANCE of the requested rate."""
        fraction = labelled_df["is_anomaly_gt"].mean()
        lo = self.ANOMALY_RATE - self.TOLERANCE
        hi = self.ANOMALY_RATE + self.TOLERANCE
        assert lo <= fraction <= hi, (
            f"Anomaly fraction {fraction:.4f} outside [{lo:.4f}, {hi:.4f}]"
        )

    def test_normal_rows_have_kind_normal(self, labelled_df):
        """Non-anomalous rows must have kind == 'normal'."""
        non_anomalous = labelled_df[~labelled_df["is_anomaly_gt"]]
        assert (non_anomalous["kind"] == "normal").all()

    def test_anomalous_rows_have_non_normal_kind(self, labelled_df):
        """Every row with is_anomaly_gt=True must have a non-'normal' kind."""
        anomalous = labelled_df[labelled_df["is_anomaly_gt"]]
        assert (anomalous["kind"] != "normal").all()


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

class TestParquetRoundTrip:
    def test_canonical_round_trip(self, tmp_path):
        """write_canonical → read_canonical preserves all TRAJECTORY_COLUMNS."""
        from src.data.synthetic import generate
        from src.data.io import write_canonical, read_canonical

        df = generate(n_agents=3, n_steps=40, seed=5)
        path = tmp_path / "rt.parquet"
        write_canonical(df, path)
        df_rt = read_canonical(path)

        # Column set
        for col in TRAJECTORY_COLUMNS:
            assert col in df_rt.columns

        # Column order
        assert list(df_rt.columns) == list(TRAJECTORY_COLUMNS)

        # Values
        df_orig = df[list(TRAJECTORY_COLUMNS)].reset_index(drop=True)
        df_rt = df_rt.reset_index(drop=True)
        pd.testing.assert_frame_equal(df_orig, df_rt)

    def test_label_columns_not_in_canonical(self, tmp_path):
        """The canonical parquet must NOT contain label columns."""
        from src.data.synthetic import generate
        from src.data.io import write_canonical, read_canonical

        df = generate(n_agents=2, n_steps=20, seed=5)
        path = tmp_path / "canon.parquet"
        write_canonical(df, path)
        df_rt = read_canonical(path)

        assert "is_anomaly_gt" not in df_rt.columns
        assert "kind" not in df_rt.columns

    def test_load_open_stub_exists(self):
        """load_open must be importable and return None or a DataFrame for a valid path stub."""
        from src.data.io import load_open
        import inspect
        assert callable(load_open)
        # Must have a path parameter
        sig = inspect.signature(load_open)
        assert len(sig.parameters) >= 1
