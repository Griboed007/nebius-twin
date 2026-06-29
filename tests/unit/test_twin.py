"""Unit tests for proposal 030 — semantic twin (src/twin/).

Tests cover:
- Read-only SPARQL guard (INSERT/DELETE/UPDATE/CONSTRUCT rejected, graph unchanged)
- Acceptance-query shapes (anomaly rollup, geofence filter)
- ASK query return shape
- Turtle round-trip via store.py

RED phase: all fail with ImportError until src/twin/ is implemented.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from contracts.schema import EnrichedPoint, EnrichedTrajectory, Prediction, TrajectoryPoint


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_enriched(
    agent_id: str,
    n_observed: int = 3,
    n_predicted: int = 2,
    anomaly_indices: tuple[int, ...] = (1, 2),
    lat_offset: float = 0.0,
) -> EnrichedTrajectory:
    """Build a minimal EnrichedTrajectory for use in tests."""
    observed = [
        EnrichedPoint(
            t=float(i),
            lat=52.0 + lat_offset + i * 0.01,
            lon=5.0 + i * 0.01,
            anomaly_score=0.9 if i in anomaly_indices else 0.1,
            is_anomaly=(i in anomaly_indices),
        )
        for i in range(n_observed)
    ]
    pred_points = [
        TrajectoryPoint(
            t=float(n_observed + j),
            lat=52.0 + lat_offset + (n_observed + j) * 0.01,
            lon=5.0 + (n_observed + j) * 0.01,
        )
        for j in range(n_predicted)
    ]
    return EnrichedTrajectory(
        agent_id=agent_id,
        observed=observed,
        prediction=Prediction(
            agent_id=agent_id,
            points=pred_points,
            anomaly_score=0.9 if anomaly_indices else 0.1,
            is_anomaly=bool(anomaly_indices),
        ),
    )


def _make_geofence_enriched() -> EnrichedTrajectory:
    """
    One agent with 2 predicted points:
      - point A at lat=60.0, lon=10.0 → OUTSIDE box [50,55] x [4,6]
      - point B at lat=52.3, lon=5.3  → INSIDE  box [50,55] x [4,6]
    Observed points are within the box (irrelevant to geofence test).
    Timestamps for predicted are disjoint from observed to avoid IRI collision.
    """
    observed = [
        EnrichedPoint(t=float(i), lat=52.0 + i * 0.01, lon=5.0 + i * 0.01,
                      anomaly_score=0.1, is_anomaly=False)
        for i in range(3)
    ]
    pred_points = [
        TrajectoryPoint(t=100.0, lat=60.0, lon=10.0),   # outside
        TrajectoryPoint(t=101.0, lat=52.3, lon=5.3),    # inside
    ]
    return EnrichedTrajectory(
        agent_id="geo_agent",
        observed=observed,
        prediction=Prediction(
            agent_id="geo_agent",
            points=pred_points,
            anomaly_score=0.1,
            is_anomaly=False,
        ),
    )


# ---------------------------------------------------------------------------
# Read-only guard tests
# ---------------------------------------------------------------------------

class TestReadOnlyGuard:
    """The sparql() function must reject mutating queries without modifying the graph."""

    def _base_graph(self):
        from src.twin.graph import build_graph
        return build_graph([_make_enriched("ro_agent")])

    def test_insert_data_rejected(self):
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "INSERT DATA { <http://a> <http://b> <http://c> }")
        assert len(g) == before

    def test_delete_data_rejected(self):
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "DELETE DATA { <http://a> <http://b> <http://c> }")
        assert len(g) == before

    def test_delete_where_rejected(self):
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "DELETE WHERE { ?s ?p ?o }")
        assert len(g) == before

    def test_insert_where_rejected(self):
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "INSERT { <http://x> <http://y> <http://z> } WHERE { ?s ?p ?o }")
        assert len(g) == before

    def test_construct_rejected(self):
        """CONSTRUCT is not SELECT/ASK and must be rejected."""
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        assert len(g) == before

    def test_describe_rejected(self):
        """DESCRIBE is not SELECT/ASK and must be rejected."""
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        with pytest.raises(QueryRejectedError):
            sparql(g, "DESCRIBE <http://example.org/twin/Agent_ro_agent>")
        assert len(g) == before

    def test_graph_unchanged_after_rejection(self):
        """Graph triple count is identical before and after a rejected query."""
        from src.twin.query import sparql, QueryRejectedError
        g = self._base_graph()
        before = len(g)
        try:
            sparql(g, "INSERT DATA { <http://a> <http://b> <http://c> }")
        except QueryRejectedError:
            pass
        assert len(g) == before

    def test_select_is_allowed(self):
        from src.twin.query import sparql
        g = self._base_graph()
        rows = sparql(g, "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
        assert isinstance(rows, list)

    def test_ask_is_allowed(self):
        from src.twin.query import sparql
        g = self._base_graph()
        rows = sparql(g, "ASK { ?s ?p ?o }")
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "_ask" in rows[0]


# ---------------------------------------------------------------------------
# ASK query return shape
# ---------------------------------------------------------------------------

class TestAskQueryShape:
    """ASK queries return [{"_ask": bool}]."""

    def test_ask_true(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        g = build_graph([_make_enriched("ask_t")])
        rows = sparql(g, "ASK { ?s ?p ?o }")
        assert rows == [{"_ask": True}]

    def test_ask_false(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        g = build_graph([_make_enriched("ask_f")])
        # Ask for a known-absent triple
        rows = sparql(
            g,
            "PREFIX ex: <http://example.org/twin/> "
            "ASK { <http://example.org/twin/NoSuchNode> ex:agentId \"x\" }",
        )
        assert rows == [{"_ask": False}]


# ---------------------------------------------------------------------------
# Acceptance query 1 — Anomaly rollup
# ---------------------------------------------------------------------------

class TestAnomalyRollup:
    """
    Acceptance query: anomaly counts per agent must match injected data.
    Query: SELECT ?a (COUNT(?o) AS ?n) WHERE {
             ?o ex:observedAgent ?a ; ex:isAnomaly true
           } GROUP BY ?a
    """

    def test_single_agent_rollup(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql, ANOMALY_ROLLUP_QUERY

        # 3 observed points, anomalies at index 1 and 2
        traj = _make_enriched("rollup_1", n_observed=4, n_predicted=1,
                               anomaly_indices=(1, 2))
        g = build_graph([traj])
        rows = sparql(g, ANOMALY_ROLLUP_QUERY)
        assert len(rows) == 1
        assert int(rows[0]["n"]) == 2

    def test_no_anomalies_returns_empty(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql, ANOMALY_ROLLUP_QUERY

        traj = _make_enriched("clean_agent", anomaly_indices=())
        g = build_graph([traj])
        rows = sparql(g, ANOMALY_ROLLUP_QUERY)
        assert rows == []

    def test_multi_agent_rollup_counts(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql, ANOMALY_ROLLUP_QUERY

        # agent_A: 2 anomalies; agent_B: 1 anomaly
        traj_a = _make_enriched("agent_A", n_observed=5, n_predicted=1,
                                 anomaly_indices=(0, 3), lat_offset=0.0)
        traj_b = _make_enriched("agent_B", n_observed=5, n_predicted=1,
                                 anomaly_indices=(2,), lat_offset=2.0)
        g = build_graph([traj_a, traj_b])
        rows = sparql(g, ANOMALY_ROLLUP_QUERY)
        counts = {str(r["a"]): int(r["n"]) for r in rows}
        assert counts["http://example.org/twin/Agent_agent_A"] == 2
        assert counts["http://example.org/twin/Agent_agent_B"] == 1

    def test_anomaly_score_stored_as_double(self):
        """anomaly_score literal is xsd:double so numeric FILTER can use it."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("score_agent", anomaly_indices=(0,))
        g = build_graph([traj])
        # FILTER on anomaly_score > 0.5 should find the anomalous point
        q = (
            "PREFIX ex: <http://example.org/twin/> "
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> "
            "SELECT ?o WHERE { "
            "  ?o ex:anomalyScore ?s . "
            "  FILTER(?s > 0.5) "
            "}"
        )
        rows = sparql(g, q)
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Acceptance query 2 — Geofence (predicted outside box)
# ---------------------------------------------------------------------------

class TestGeofenceFilter:
    """Predicted points OUTSIDE a lat/lon box are returned."""

    def test_outside_box_returned(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql, geofence_query

        g = build_graph([_make_geofence_enriched()])
        # Box: lat [50, 55], lon [4, 6]
        q = geofence_query(lat_min=50.0, lat_max=55.0, lon_min=4.0, lon_max=6.0)
        rows = sparql(g, q)
        lats = [float(r["lat"]) for r in rows]
        assert 60.0 in lats

    def test_inside_box_excluded(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql, geofence_query

        g = build_graph([_make_geofence_enriched()])
        q = geofence_query(lat_min=50.0, lat_max=55.0, lon_min=4.0, lon_max=6.0)
        rows = sparql(g, q)
        lats = [float(r["lat"]) for r in rows]
        # inside point (lat=52.3) must NOT appear
        assert not any(abs(la - 52.3) < 0.001 for la in lats)

    def test_observed_excluded_from_geofence(self):
        """Observed points are never returned by the geofence query (kind=predicted only).

        Use a box that covers all predicted points — the query selects only
        kind='predicted' points OUTSIDE the box, so the result is empty.
        Observed points are excluded by the ex:kind filter, not by coordinates.
        """
        from src.twin.graph import build_graph
        from src.twin.query import sparql, geofence_query

        # Small, fixed trajectory: observed at lat~52, predicted at lat~52+
        traj = _make_enriched("obs_excl", n_observed=2, n_predicted=2,
                               anomaly_indices=(), lat_offset=0.0)
        g = build_graph([traj])
        # Box contains all the predicted points (lat 52.0x, lon 5.0x)
        q = geofence_query(lat_min=50.0, lat_max=55.0, lon_min=4.0, lon_max=6.0)
        # All predicted points are INSIDE the box → geofence returns empty
        rows = sparql(g, q)
        assert rows == []


# ---------------------------------------------------------------------------
# Turtle round-trip
# ---------------------------------------------------------------------------

class TestTurtleRoundTrip:
    """Serialise → load → isomorphic with the original graph."""

    def test_round_trip_isomorphic(self):
        from rdflib.compare import isomorphic
        from src.twin.graph import build_graph
        from src.twin.store import save_turtle, load_turtle

        trajs = [_make_enriched(f"rt_{i}", lat_offset=i * 0.5) for i in range(3)]
        g1 = build_graph(trajs)

        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
            path = pathlib.Path(f.name)

        save_turtle(g1, path)
        g2 = load_turtle(path)
        path.unlink(missing_ok=True)

        assert isomorphic(g1, g2), "Loaded graph is not isomorphic to original"

    def test_round_trip_orphan_check(self):
        """Orphan check is still empty after a round-trip through Turtle.

        Uses the canonical ORPHAN_CHECK_QUERY constant from query.py so that
        any drift between the stored query and what was tested is caught here.
        Also checks non-vacuously: the loaded graph must actually contain
        saref:Observation nodes before we assert zero orphans.
        """
        from src.twin.graph import build_graph
        from src.twin.query import sparql, ORPHAN_CHECK_QUERY
        from src.twin.store import save_turtle, load_turtle

        # n_observed=3, n_predicted=2 → 5 Observation nodes in the loaded graph
        g1 = build_graph([_make_enriched("persist", n_observed=3, n_predicted=2)])

        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
            path = pathlib.Path(f.name)

        save_turtle(g1, path)
        g2 = load_turtle(path)
        path.unlink(missing_ok=True)

        # Non-vacuous: confirm Observation nodes survived the round-trip
        q_count = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT (COUNT(?o) AS ?n) WHERE { ?o a saref:Observation }"
        )
        count_rows = sparql(g2, q_count)
        assert int(count_rows[0]["n"]) == 5, "Observation nodes lost during Turtle round-trip"

        rows = sparql(g2, ORPHAN_CHECK_QUERY)
        assert rows == []

    def test_save_creates_file(self):
        from src.twin.graph import build_graph
        from src.twin.store import save_turtle

        g = build_graph([_make_enriched("file_agent")])
        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as f:
            path = pathlib.Path(f.name)

        save_turtle(g, path)
        assert path.exists()
        content = path.read_text()
        # The Turtle file must reference the saref namespace and Observation type (v2 contract)
        assert "saref" in content and "Observation" in content
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Structural / triple shape tests
# ---------------------------------------------------------------------------

class TestTripleShapes:
    """Triples emitted by build_graph conform exactly to saref_mapping.md."""

    def test_agent_id_triple(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("shape_ag")
        g = build_graph([traj])
        q = (
            "PREFIX ex: <http://example.org/twin/> "
            "SELECT ?id WHERE { "
            "  <http://example.org/twin/Agent_shape_ag> ex:agentId ?id "
            "}"
        )
        rows = sparql(g, q)
        assert len(rows) == 1
        assert str(rows[0]["id"]) == "shape_ag"

    def test_has_timestamp_triple(self):
        """Each measurement has a saref:hasTimestamp literal."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("ts_agent", n_observed=1, n_predicted=0)
        g = build_graph([traj])
        q = (
            "PREFIX saref: <https://saref.etsi.org/core/> "
            "SELECT ?ts WHERE { ?o saref:hasTimestamp ?ts }"
        )
        rows = sparql(g, q)
        assert len(rows) == 1
        val = str(rows[0]["ts"])
        # Must be ISO8601-style (contains 'T' or 'Z' or '+')
        assert "T" in val or "Z" in val or "+" in val

    def test_lat_lon_are_double(self):
        """geo:lat and geo:long are stored as xsd:double, supporting numeric FILTER."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("latlon_agent", n_observed=1, n_predicted=0)
        g = build_graph([traj])
        q = (
            "PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> "
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> "
            "SELECT ?lat ?lon WHERE { "
            "  ?o geo:lat ?lat ; geo:long ?lon . "
            "  FILTER(?lat > 40.0) "
            "}"
        )
        rows = sparql(g, q)
        assert len(rows) == 1
        assert float(rows[0]["lat"]) > 40.0

    def test_is_anomaly_is_boolean(self):
        """ex:isAnomaly is stored as xsd:boolean — the rollup query relies on this."""
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        # anomaly_indices=(0,) → first point is anomalous
        traj = _make_enriched("bool_agent", n_observed=2, n_predicted=0, anomaly_indices=(0,))
        g = build_graph([traj])
        # SPARQL: match the bare `true` boolean literal
        q = (
            "PREFIX ex: <http://example.org/twin/> "
            "SELECT ?o WHERE { ?o ex:isAnomaly true }"
        )
        rows = sparql(g, q)
        assert len(rows) == 1

    def test_kind_observed_and_predicted(self):
        from src.twin.graph import build_graph
        from src.twin.query import sparql

        traj = _make_enriched("kind_agent", n_observed=2, n_predicted=2)
        g = build_graph([traj])
        q_obs = (
            "PREFIX ex: <http://example.org/twin/> "
            'SELECT ?o WHERE { ?o ex:kind "observed" }'
        )
        q_pred = (
            "PREFIX ex: <http://example.org/twin/> "
            'SELECT ?o WHERE { ?o ex:kind "predicted" }'
        )
        assert len(sparql(g, q_obs)) == 2
        assert len(sparql(g, q_pred)) == 2
